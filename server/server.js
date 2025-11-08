import express from "express";
import cors from "cors";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { spawn } from "child_process";
import duckdb from "duckdb";
import ExcelJS from "exceljs";
import PDFDocument from "pdfkit";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// 👇 points to the existing root-level data folder
const DATA_DIR = path.join(__dirname, "../data");

// optional static payload (if you put a copy here)
const THEMES_PAYLOAD = path.join(__dirname, "../frontend/public/themes_payload.json");

// parquet + names/summaries in root/data
const THEMES_PARQUET = path.join(DATA_DIR, "tweets_stage3_themes.parquet");
const THEMES_STAGE2_DEFAULT = path.join(DATA_DIR, "tweets_stage2_aspects.parquet");
const THEMES_STAGE2_OVERRIDE = process.env.THEMES_PARQUET_PATH
  ? path.resolve(process.env.THEMES_PARQUET_PATH)
  : null;
const THEMES_MAX_ROWS = (() => {
  if (process.env.THEMES_MAX_ROWS !== undefined) {
    const parsed = parseInt(process.env.THEMES_MAX_ROWS, 10);
    return Number.isFinite(parsed) && parsed > 0 ? parsed : 10000;
  }
  return 10000; // enforce <=10k rows by default
})();
const NORMALIZED_DATA_DIR = path.normalize(DATA_DIR).toLowerCase();
const THEME_NAMES = path.join(DATA_DIR, "theme_names.json");
const THEME_SUMMARIES = path.join(DATA_DIR, "theme_summaries.json");
const ASPECT_FALLBACK_LABEL = "Other";

const app = express();
app.use(cors());
app.use(express.json());

const db = new duckdb.Database(":memory:");
const conn = db.connect();

// Root route
app.get("/", async (req, res) => {
  try {
    // Try to get date range from parquet file
    let dateRange = null;
    if (exists(THEMES_PARQUET)) {
      try {
        // Get min/max dates from timestamp column
        // The createdat column is a timestamp, so we can use DATE() directly
        const sql = `
          SELECT DISTINCT
            DATE(createdat) as date_val
          FROM read_parquet('${THEMES_PARQUET.replace(/\\/g, "/")}')
          ORDER BY date_val;
        `;
        const rows = await new Promise((resolve, reject) =>
          conn.all(sql, (err, r) => (err ? reject(err) : resolve(r)))
        );
        if (rows && rows.length > 0) {
          // Get first and last dates from the sorted list
          const dates = rows.map(row => {
            let date = row.date_val;
            if (typeof date === 'string') {
              date = new Date(date);
            }
            return date instanceof Date && !isNaN(date.getTime()) ? date : null;
          }).filter(d => d !== null).sort((a, b) => a - b);
          
          if (dates.length > 0) {
            const minDate = dates[0];
            const maxDate = dates[dates.length - 1];
            const minStr = minDate.toISOString().split('T')[0];
            const maxStr = maxDate.toISOString().split('T')[0];
            dateRange = {
              min: minStr,
              max: maxStr
            };
          }
        }
      } catch (e) {
        console.error("Error getting date range:", e);
      }
    }
    
    res.json({
      message: "Social Listening API",
      version: "1.0.0",
      date_range: dateRange || null, // No fallback - return null if dates can't be determined
      endpoints: {
        themes: "GET /api/themes",
        themeTweets: "GET /api/themes/:id/tweets?limit=20&q=search"
      }
    });
  } catch (e) {
    res.json({
      message: "Social Listening API",
      version: "1.0.0",
      date_range: null,
      endpoints: {
        themes: "GET /api/themes",
        themeTweets: "GET /api/themes/:id/tweets?limit=20&q=search"
      }
    });
  }
});

const exists = (p) => { try { fs.accessSync(p); return true; } catch { return false; } };

// Helper function to serialize rows (convert BigInt to string)
function serializeRows(rows) {
  return rows.map(row => {
    const obj = {};
    for (const [key, value] of Object.entries(row)) {
      if (typeof value === 'bigint') {
        obj[key] = value.toString();
      } else if (value instanceof Date) {
        obj[key] = value.toISOString().split('T')[0];
      } else {
        obj[key] = value;
      }
    }
    if (Object.prototype.hasOwnProperty.call(obj, "aspect_dominant")) {
      obj.aspect_dominant = normalizeAspectName(obj.aspect_dominant);
    }
    return obj;
  });
}

function normalizeAspectName(value) {
  const text = (value ?? "").toString().trim();
  if (!text) return ASPECT_FALLBACK_LABEL;
  const lower = text.toLowerCase();
  if (lower === "none" || lower === "unspecified") {
    return ASPECT_FALLBACK_LABEL;
  }
  return text;
}

function toDatabaseAspect(value) {
  if (value == null) return null;
  const text = value.toString().trim();
  if (!text) return null;
  if (text.toLowerCase() === ASPECT_FALLBACK_LABEL.toLowerCase()) {
    return "none";
  }
  return text.toLowerCase().replace(/\s+/g, "_").replace(/\//g, "_");
}

// Helper function to build WHERE clause with date filtering
function buildWhereClause(start, end, additionalConditions = []) {
  const conditions = [];
  
  // Only add date filters if both start and end are provided and valid
  // The createdat column is a timestamp, so we can use DATE() directly
  if (start && end && start.trim() !== "" && end.trim() !== "") {
    const startStr = start.trim();
    const endStr = end.trim();
    
    // Handle both timestamp and string date formats
    // Try to parse as timestamp first, fallback to string parsing
    conditions.push(`DATE(createdat) >= DATE('${startStr}')`);
    conditions.push(`DATE(createdat) <= DATE('${endStr}')`);
  }
  
  conditions.push(...additionalConditions);
  return conditions.length > 0 ? `WHERE ${conditions.join(" AND ")}` : "";
}

// List themes (dynamically generates themes using Python clustering - takes 2+ minutes)
app.get("/api/themes", async (req, res) => {
  try {
    const start = req.query.start || null;
    const end = req.query.end || null;
    const n_clusters = req.query.n_clusters ? parseInt(req.query.n_clusters) : 12; // Default 12 professional themes

    // Resolve parquet path: query param > env override > default
    let parquetPath = THEMES_STAGE2_DEFAULT;
    if (THEMES_STAGE2_OVERRIDE) {
      const overrideNormalized = path.normalize(THEMES_STAGE2_OVERRIDE).toLowerCase();
      if (overrideNormalized.startsWith(NORMALIZED_DATA_DIR)) {
        parquetPath = THEMES_STAGE2_OVERRIDE;
      } else {
        console.warn(`[Themes API] ⚠️ Ignoring THEMES_PARQUET_PATH outside data directory: ${THEMES_STAGE2_OVERRIDE}`);
      }
    }

    if (req.query.parquet) {
      const candidate = path.resolve(DATA_DIR, req.query.parquet);
      const candidateNormalized = path.normalize(candidate).toLowerCase();
      if (!candidateNormalized.startsWith(NORMALIZED_DATA_DIR)) {
        return res.status(400).json({
          error: "Invalid parquet path",
          details: "Custom parquet must reside within the data directory",
        });
      }
      parquetPath = candidate;
    }

    const maxRowsParam = req.query.max_rows ? parseInt(req.query.max_rows, 10) : null;
    const maxRows = Number.isFinite(maxRowsParam) && maxRowsParam > 0
      ? maxRowsParam
      : THEMES_MAX_ROWS;

    if (!exists(parquetPath)) {
      return res.status(404).json({ 
        error: "Stage 2 parquet file not found",
        message: "Please ensure the parquet file exists before generating themes",
        details: parquetPath
      });
    }

    console.log(`[Themes API] 🚀 Generating themes dynamically... (n_clusters=${n_clusters}, start=${start}, end=${end}, parquet=${parquetPath}${maxRows ? `, max_rows=${maxRows}` : ""})`);
    console.log(`[Themes API] ⏱️  This may take 2-3 minutes...`);
    
    // Generate themes dynamically using Python script
    const scriptPath = path.join(__dirname, "..", "scripts", "generate_themes_api.py");
    
    // Check if script exists
    if (!exists(scriptPath)) {
      console.error(`[Themes API] ❌ Python script not found at: ${scriptPath}`);
      return res.status(500).json({ 
        error: "Theme generation script not found",
        details: `The script 'generate_themes_api.py' is missing from the scripts directory`,
        hint: "Ensure the script exists at: " + scriptPath
      });
    }
    
    const args = [
      scriptPath,
      "--parquet", parquetPath,
      "--n-clusters", String(n_clusters)
    ];

    if (maxRows) {
      args.push("--max-rows", String(maxRows));
    }
    
    if (start) args.push("--start-date", start);
    if (end) args.push("--end-date", end);
    
    // Get OpenAI key from env if available
    const openaiKey = process.env.OPENAI_API_KEY;
    if (openaiKey) {
      args.push("--openai-key", openaiKey);
    }
    
    // Determine Python command (python3 on Unix, python on Windows)
    const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
    
    const pythonProcess = spawn(pythonCmd, args, {
      cwd: path.join(__dirname, ".."),
      stdio: ['pipe', 'pipe', 'pipe'],
      shell: process.platform === 'win32' // Use shell on Windows for better compatibility
    });
    
    let stdout = '';
    let stderr = '';
    
    pythonProcess.stdout.on('data', (data) => {
      stdout += data.toString();
    });
    
    pythonProcess.stderr.on('data', (data) => {
      stderr += data.toString();
      // Only log to console, don't treat as error yet (Python may write warnings to stderr)
      process.stdout.write(data);
    });
    
    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        console.error(`[Themes API] ❌ Python script exited with code ${code}`);
        console.error(`[Themes API] stderr: ${stderr}`);
        console.error(`[Themes API] stdout: ${stdout.substring(0, 500)}`);
        
        // Try to parse error from stdout if it's JSON
        let errorDetails = stderr || "Unknown error";
        let errorHint = "Check Python dependencies and ensure tweets_stage2_aspects.parquet exists";
        
        try {
          const errorPayload = JSON.parse(stdout);
          if (errorPayload.error) {
            errorDetails = errorPayload.error;
            if (errorPayload.type) {
              errorDetails += ` (${errorPayload.type})`;
            }
          }
        } catch (e) {
          // stdout is not JSON, use stderr or default message
          if (stdout.trim()) {
            errorDetails = stdout.trim().substring(0, 200);
          }
        }
        
        if (!res.headersSent) {
          return res.status(500).json({ 
            error: "Theme generation failed",
            details: errorDetails,
            exitCode: code,
            hint: errorHint
          });
        }
        return;
      }
      
      try {
        // Parse JSON output from Python script
        const payload = JSON.parse(stdout);
        
        if (payload.error) {
          if (!res.headersSent) {
            return res.status(500).json({ 
              error: payload.error, 
              type: payload.type,
              details: payload.details || stderr || "No additional details available"
            });
          }
          return;
        }
        
        console.log(`[Themes API] ✅ Success! Generated ${payload.themes?.length || 0} themes`);
        
        // Return the generated themes
        if (!res.headersSent) {
          res.json({
            themes: payload.themes || [],
            updated_at: payload.updated_at || new Date().toISOString(),
            used_llm: payload.used_llm || false
          });
        }
      } catch (parseError) {
        console.error(`[Themes API] ❌ Failed to parse Python output:`, parseError);
        console.error(`[Themes API] stdout (first 500 chars):`, stdout.substring(0, 500));
        console.error(`[Themes API] stderr:`, stderr.substring(0, 500));
        
        if (!res.headersSent) {
          return res.status(500).json({ 
            error: "Failed to parse theme generation output",
            details: parseError.message,
            stdout_preview: stdout.substring(0, 200),
            stderr_preview: stderr.substring(0, 200),
            hint: "The Python script may have encountered an error. Check server logs for details."
          });
        }
      }
    });
    
    pythonProcess.on('error', (err) => {
      console.error(`[Themes API] ❌ Failed to spawn Python process:`, err);
      if (!res.headersSent) {
        return res.status(500).json({ 
          error: "Failed to start theme generation",
          details: err.message,
          hint: `Make sure Python is installed and accessible as '${pythonCmd}'. Check PATH.`
        });
      }
    });
    
    // Set timeout to prevent hanging (10 minutes for theme generation with 12 themes + OpenAI)
    setTimeout(() => {
      if (!res.headersSent) {
        pythonProcess.kill();
        res.status(504).json({ 
          error: "Theme generation timeout",
          details: "Theme generation exceeded 10 minutes. This may happen with large datasets or when generating many themes.",
          hint: "Try reducing the number of themes or the date range."
        });
      }
    }, 10 * 60 * 1000); // 10 minutes for 12 themes with OpenAI
    
  } catch (e) {
    console.error(e);
    if (!res.headersSent) {
      res.status(500).json({ error: String(e) });
    }
  }
});

// Drill-down tweets for a theme
app.get("/api/themes/:id/tweets", async (req, res) => {
  try {
    // Check if parquet file exists
    if (!exists(THEMES_PARQUET)) {
      return res.status(404).json({
        error: "Parquet file not found",
        message: `The required data file 'tweets_stage3_themes.parquet' is missing from the data directory.`,
        details: `Please generate the parquet file by running the theme generation pipeline. The file should be located at: ${THEMES_PARQUET}`,
        suggestion: "Run the theme generation script to create the required parquet file."
      });
    }

    const themeId = Number(req.params.id);
    const limit = Math.min(Number(req.query.limit || 20), 200);
    const q = (req.query.q || "").toString().trim();

    const defaultFields = [
      "id", "twitterurl",
      "text_clean", "text", "fulltext",
      "sentiment_label", "sentiment_score",
      "aspect_pricing", "aspect_delivery", "aspect_returns", "aspect_staff",
      "aspect_app_ux",
      "aspect_dominant",
      "createdat", "lang", "has_url", "has_hashtag"
    ];
    const fields = (req.query.fields ? req.query.fields.split(",") : defaultFields).join(", ");

    const conditions = [`theme = ${themeId}`];
    if (req.query.start) conditions.push(`createdat >= '${req.query.start}'`);
    if (req.query.end) conditions.push(`createdat <= '${req.query.end}'`);
    if (q) {
      const like = `%${q.replace(/'/g, "''")}%`;
      conditions.push(`(text_clean ILIKE '${like}' OR text ILIKE '${like}' OR fulltext ILIKE '${like}')`);
    }

    const sql = `
      SELECT ${fields}
      FROM read_parquet('${THEMES_PARQUET.replace(/\\/g, "/")}')
      WHERE ${conditions.join(" AND ")}
      ORDER BY createdat DESC
      LIMIT ${limit};
    `;
    const rows = await new Promise((resolve, reject) =>
      conn.all(sql, (err, r) => (err ? reject(err) : resolve(r)))
    );

    // Convert BigInt values to strings for JSON serialization
    const serializedRows = serializeRows(rows);

    res.json({ theme: themeId, count: rows.length, items: serializedRows });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: String(e) });
  }
});

// Sentiment Summary
app.get("/api/sentiment/summary", async (req, res) => {
  try {
    if (!exists(THEMES_PARQUET)) {
      return res.status(404).json({ 
        error: "Parquet file not found",
        details: `The required data file 'tweets_stage3_themes.parquet' is missing from the data directory.`,
        hint: "Please ensure the parquet file exists at: " + THEMES_PARQUET
      });
    }

    const start = req.query.start;
    const end = req.query.end;
    const whereClause = buildWhereClause(start, end);

    const sql = `
      SELECT 
        sentiment_label,
        COUNT(*)::INTEGER as count,
        AVG(sentiment_score)::DOUBLE as avg_score
      FROM read_parquet('${THEMES_PARQUET.replace(/\\/g, "/")}')
      ${whereClause}
      GROUP BY sentiment_label;
    `;
    
    console.log(`[Sentiment Summary] Query: ${sql.substring(0, 200)}...`);
    
    const rows = await new Promise((resolve, reject) =>
      conn.all(sql, (err, r) => {
        if (err) {
          console.error(`[Sentiment Summary] SQL Error:`, err);
          reject(err);
        } else {
          resolve(r);
        }
      })
    );

    const summary = {
      positive: 0,
      negative: 0,
      neutral: 0,
      total: 0,
      avg_scores: {},
      counts: { positive: 0, negative: 0, neutral: 0 },
      percent: { positive: 0, negative: 0, neutral: 0 }
    };

    rows.forEach(row => {
      const label = row.sentiment_label?.toLowerCase() || 'neutral';
      const count = Number(row.count) || 0;
      summary[label] = count;
      summary.counts[label] = count;
      summary.total += count;
      if (row.avg_score) {
        summary.avg_scores[label] = Number(row.avg_score);
      }
    });

    // Calculate percentages
    if (summary.total > 0) {
      summary.percent.positive = Number(((summary.positive / summary.total) * 100).toFixed(1));
      summary.percent.negative = Number(((summary.negative / summary.total) * 100).toFixed(1));
      summary.percent.neutral = Number(((summary.neutral / summary.total) * 100).toFixed(1));
    }

    res.json(summary);
  } catch (e) {
    console.error(`[Sentiment Summary] Error:`, e);
    res.status(500).json({ 
      error: "Failed to load sentiment summary",
      details: e.message || String(e),
      hint: "Check if the parquet file exists and has the correct schema. Verify date format matches 'Aug-14-2025'."
    });
  }
});

// Sentiment Trend
app.get("/api/sentiment/trend", async (req, res) => {
  try {
    if (!exists(THEMES_PARQUET)) {
      return res.status(404).json({ 
        error: "Parquet file not found",
        details: `The required data file 'tweets_stage3_themes.parquet' is missing from the data directory.`,
        hint: "Please ensure the parquet file exists at: " + THEMES_PARQUET
      });
    }

    const start = req.query.start;
    const end = req.query.end;
    const period = req.query.period || "daily";
    const offset = Number(req.query.offset || 0);
    const limitParam = Number(req.query.limit ?? 0);

    const whereClause = buildWhereClause(start, end);
    
    // The createdat column is a timestamp, so we can use DATE() directly
    // No need for STRPTIME since it's already a date/timestamp type
    let dateFormat = "DATE(createdat)";
    if (period === "weekly") {
      dateFormat = "DATE_TRUNC('week', DATE(createdat))";
    } else if (period === "monthly") {
      dateFormat = "DATE_TRUNC('month', DATE(createdat))";
    }

    const limitClause = limitParam > 0 ? `LIMIT ${limitParam}` : "";
    const offsetClause = limitParam > 0 && offset > 0 ? `OFFSET ${offset}` : "";

    const sql = `
      SELECT 
        ${dateFormat} as date,
        sentiment_label,
        COUNT(*)::INTEGER as count
      FROM read_parquet('${THEMES_PARQUET.replace(/\\/g, "/")}')
      ${whereClause}
      GROUP BY ${dateFormat}, sentiment_label
      ORDER BY date DESC
      ${limitClause} ${offsetClause};
    `;
    
    console.log(`[Sentiment Trend] Query: ${sql.substring(0, 200)}...`);
    
    const rows = await new Promise((resolve, reject) =>
      conn.all(sql, (err, r) => {
        if (err) {
          console.error(`[Sentiment Trend] SQL Error:`, err);
          reject(err);
        } else {
          resolve(r);
        }
      })
    );

    // Group by date and calculate totals and percentages
    const trendMap = {};
    rows.forEach(row => {
      const date = row.date instanceof Date 
        ? row.date.toISOString().split('T')[0]
        : String(row.date).split('T')[0];
      if (!trendMap[date]) {
        trendMap[date] = { date, positive: 0, negative: 0, neutral: 0, total: 0 };
      }
      const label = row.sentiment_label?.toLowerCase() || 'neutral';
      const count = Number(row.count) || 0;
      trendMap[date][label] = count;
      trendMap[date].total += count;
    });

    // Convert counts to percentages for the graph
    const trend = Object.values(trendMap).map(item => {
      const total = item.total || 1; // Avoid division by zero
      return {
        date: item.date,
        positive: total > 0 ? Number(((item.positive / total) * 100).toFixed(1)) : 0,
        negative: total > 0 ? Number(((item.negative / total) * 100).toFixed(1)) : 0,
        neutral: total > 0 ? Number(((item.neutral / total) * 100).toFixed(1)) : 0,
        // Also include counts for reference
        counts: {
          positive: item.positive,
          negative: item.negative,
          neutral: item.neutral,
          total: item.total
        }
      };
    }).sort((a, b) => a.date.localeCompare(b.date));

    res.json({ trend, period, offset: limitParam > 0 ? offset : 0, limit: limitParam });
  } catch (e) {
    console.error(`[Sentiment Trend] Error:`, e);
    res.status(500).json({ 
      error: "Failed to load sentiment trend",
      details: e.message || String(e),
      hint: "Check if the parquet file exists and has the correct schema. Verify date format matches 'Aug-14-2025'."
    });
  }
});

// Aspects Summary
app.get("/api/aspects/summary", async (req, res) => {
  try {
    if (!exists(THEMES_PARQUET)) {
      return res.status(404).json({ error: "Parquet file not found" });
    }

    const start = req.query.start;
    const end = req.query.end;
    const asPercent = req.query.as_percent === 'true';
    const whereClause = buildWhereClause(start, end);

    const sql = `
      SELECT 
        aspect_dominant,
        COUNT(*)::INTEGER as count
      FROM read_parquet('${THEMES_PARQUET.replace(/\\/g, "/")}')
      ${whereClause}
      GROUP BY aspect_dominant
      ORDER BY count DESC;
    `;
    
    const rows = await new Promise((resolve, reject) =>
      conn.all(sql, (err, r) => (err ? reject(err) : resolve(r)))
    );

    const total = rows.reduce((sum, row) => sum + Number(row.count), 0);
    const summary = {};

    rows.forEach(row => {
      const aspect = normalizeAspectName(row.aspect_dominant);
      const count = Number(row.count) || 0;
      summary[aspect] = asPercent ? (total > 0 ? (count / total * 100).toFixed(2) : 0) : count;
    });

    res.json(summary);
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: String(e) });
  }
});

// Aspect Average Scores
app.get("/api/aspects/avg-scores", async (req, res) => {
  try {
    if (!exists(THEMES_PARQUET)) {
      return res.status(404).json({ error: "Parquet file not found" });
    }

    const start = req.query.start;
    const end = req.query.end;
    const whereClause = buildWhereClause(start, end);

    const sql = `
      SELECT 
        AVG(aspect_pricing)::DOUBLE as pricing,
        AVG(aspect_delivery)::DOUBLE as delivery,
        AVG(aspect_returns)::DOUBLE as returns,
        AVG(aspect_staff)::DOUBLE as staff,
        AVG(aspect_app_ux)::DOUBLE as app_ux
      FROM read_parquet('${THEMES_PARQUET.replace(/\\/g, "/")}')
      ${whereClause};
    `;
    
    const rows = await new Promise((resolve, reject) =>
      conn.all(sql, (err, r) => (err ? reject(err) : resolve(r)))
    );

    if (rows && rows.length > 0) {
      res.json({
        pricing: Number(rows[0].pricing) || 0,
        delivery: Number(rows[0].delivery) || 0,
        returns: Number(rows[0].returns) || 0,
        staff: Number(rows[0].staff) || 0,
        app_ux: Number(rows[0].app_ux) || 0
      });
    } else {
      res.json({ pricing: 0, delivery: 0, returns: 0, staff: 0, app_ux: 0 });
    }
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: String(e) });
  }
});

// Aspect × Sentiment Split
app.get("/api/aspects/sentiment-split", async (req, res) => {
  try {
    if (!exists(THEMES_PARQUET)) {
      return res.status(404).json({ error: "Parquet file not found" });
    }

    const start = req.query.start;
    const end = req.query.end;
    const asPercent = req.query.as_percent === 'true';
    const includeOthers = req.query.include_others === 'true';
    const whereClause = buildWhereClause(start, end);

    const sql = `
      SELECT 
        aspect_dominant,
        sentiment_label,
        COUNT(*)::INTEGER as count
      FROM read_parquet('${THEMES_PARQUET.replace(/\\/g, "/")}')
      ${whereClause}
      GROUP BY aspect_dominant, sentiment_label
      ORDER BY aspect_dominant, sentiment_label;
    `;
    
    const rows = await new Promise((resolve, reject) =>
      conn.all(sql, (err, r) => (err ? reject(err) : resolve(r)))
    );

    // Group by aspect
    const aspectMap = {};
    rows.forEach(row => {
      const aspect = normalizeAspectName(row.aspect_dominant);
      const sentiment = row.sentiment_label?.toLowerCase() || 'neutral';
      const count = Number(row.count) || 0;
      
      if (!aspectMap[aspect]) {
        aspectMap[aspect] = { aspect, positive: 0, negative: 0, neutral: 0, total: 0 };
      }
      aspectMap[aspect][sentiment] = count;
      aspectMap[aspect].total += count;
    });

    // Filter out fallback aspect if not including others
    let aspects = Object.values(aspectMap);
    if (!includeOthers) {
      aspects = aspects.filter(item => item.aspect !== ASPECT_FALLBACK_LABEL);
    }

    // Sort by total count descending
    aspects.sort((a, b) => b.total - a.total);

    // Transform to frontend-expected format: { labels, counts: { positive, negative, neutral }, percent?: { positive, negative, neutral } }
    const labels = aspects.map(item => item.aspect);
    const counts = {
      positive: aspects.map(item => item.positive),
      negative: aspects.map(item => item.negative),
      neutral: aspects.map(item => item.neutral)
    };

    const result = {
      labels,
      counts
    };

    // Add percentages if requested
    if (asPercent) {
      result.percent = {
        positive: aspects.map(item => item.total > 0 ? Number(((item.positive / item.total) * 100).toFixed(1)) : 0),
        negative: aspects.map(item => item.total > 0 ? Number(((item.negative / item.total) * 100).toFixed(1)) : 0),
        neutral: aspects.map(item => item.total > 0 ? Number(((item.neutral / item.total) * 100).toFixed(1)) : 0)
      };
    }

    res.json(result);
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: String(e) });
  }
});

// Sample Tweets
app.get("/api/tweets/sample", async (req, res) => {
  try {
    if (!exists(THEMES_PARQUET)) {
      return res.status(404).json({ error: "Parquet file not found" });
    }

    const start = req.query.start;
    const end = req.query.end;
    const aspect = req.query.aspect;
    const sentiment = req.query.sentiment;
    const limit = Math.min(Number(req.query.limit || 10), 100);

    // Use the same date filtering approach as other endpoints
    // The createdat column is a timestamp, so we can use DATE() directly
    const dateConditions = [];
    if (start && end && start.trim() !== "" && end.trim() !== "") {
      const startStr = start.trim();
      const endStr = end.trim();
      dateConditions.push(`DATE(createdat) >= DATE('${startStr}')`);
      dateConditions.push(`DATE(createdat) <= DATE('${endStr}')`);
    }
    
    const conditions = [...dateConditions];
    const dbAspect = toDatabaseAspect(aspect);
    if (dbAspect) {
      conditions.push(`aspect_dominant = '${dbAspect.replace(/'/g, "''")}'`);
    }
    if (sentiment) {
      conditions.push(`LOWER(sentiment_label) = '${sentiment.toLowerCase()}'`);
    }

    const whereClause = conditions.length > 0 ? `WHERE ${conditions.join(" AND ")}` : "";

    const sql = `
      SELECT 
        id, twitterurl, text_clean, text, fulltext,
        sentiment_label, sentiment_score, aspect_dominant, createdat
      FROM read_parquet('${THEMES_PARQUET.replace(/\\/g, "/")}')
      ${whereClause}
      ORDER BY createdat DESC
      LIMIT ${limit};
    `;
    
    const rows = await new Promise((resolve, reject) =>
      conn.all(sql, (err, r) => (err ? reject(err) : resolve(r)))
    );

    // Return both full tweet objects and text strings for compatibility
    const serializedRows = serializeRows(rows);
    const tweets = serializedRows.map(tweet => {
      // Return text_clean if available, otherwise text, otherwise fulltext
      return tweet.text_clean || tweet.text || tweet.fulltext || '';
    });

    res.json({ tweets, tweetObjects: serializedRows });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: String(e) });
  }
});

// Executive Summary - Enhanced with data-driven insights
app.get("/api/executive-summary", async (req, res) => {
  try {
    if (!exists(THEMES_PARQUET)) {
      return res.status(404).json({ error: "Parquet file not found" });
    }

    const start = req.query.start;
    const end = req.query.end;
    const samplePerSentiment = Number(req.query.sample_per_sentiment) || 250;
    const whereClause = buildWhereClause(start, end);

    // Get summary statistics
    const sql = `
      SELECT 
        COUNT(*)::INTEGER as total,
        COUNT(CASE WHEN sentiment_label = 'positive' THEN 1 END)::INTEGER as positive,
        COUNT(CASE WHEN sentiment_label = 'negative' THEN 1 END)::INTEGER as negative,
        COUNT(CASE WHEN sentiment_label = 'neutral' THEN 1 END)::INTEGER as neutral,
        AVG(sentiment_score)::DOUBLE as avg_sentiment
      FROM read_parquet('${THEMES_PARQUET.replace(/\\/g, "/")}')
      ${whereClause};
    `;
    
    const rows = await new Promise((resolve, reject) =>
      conn.all(sql, (err, r) => (err ? reject(err) : resolve(r)))
    );

    const stats = rows[0];
    const total = Number(stats.total) || 0;
    const positive = Number(stats.positive) || 0;
    const negative = Number(stats.negative) || 0;
    const neutral = Number(stats.neutral) || 0;
    
    // Get aspect breakdown
    const aspectSql = `
      SELECT 
        aspect_dominant,
        COUNT(*)::INTEGER as count
      FROM read_parquet('${THEMES_PARQUET.replace(/\\/g, "/")}')
      ${whereClause}
      GROUP BY aspect_dominant
      ORDER BY count DESC
      LIMIT 5;
    `;
    
    const aspectRows = await new Promise((resolve, reject) =>
      conn.all(aspectSql, (err, r) => (err ? reject(err) : resolve(r)))
    );

    // Generate data-driven executive summary
    const positivePct = total > 0 ? ((positive / total) * 100).toFixed(1) : 0;
    const negativePct = total > 0 ? ((negative / total) * 100).toFixed(1) : 0;
    const neutralPct = total > 0 ? ((neutral / total) * 100).toFixed(1) : 0;
    
    const topAspects = aspectRows
      .map(row => normalizeAspectName(row.aspect_dominant))
      .filter(aspect => aspect && aspect !== ASPECT_FALLBACK_LABEL)
      .slice(0, 3);

    // Generate summary text based on data
    let summaryText = `Executive Summary: Analysis of ${total.toLocaleString()} social media conversations from ${start} to ${end}.\n\n`;
    
    summaryText += `Sentiment Overview:\n`;
    summaryText += `The data reveals a ${negativePct > 40 ? 'predominantly negative' : neutralPct > 40 ? 'largely neutral' : 'mixed'} sentiment landscape. `;
    summaryText += `${negativePct}% of conversations express negative sentiment, while ${positivePct}% are positive and ${neutralPct}% are neutral. `;
    summaryText += `The average sentiment score of ${Number(stats.avg_sentiment || 0).toFixed(2)} indicates ${Number(stats.avg_sentiment || 0) > 0.7 ? 'moderately positive' : Number(stats.avg_sentiment || 0) > 0.5 ? 'neutral' : 'negative'} overall sentiment.\n\n`;
    
    if (topAspects.length > 0) {
      summaryText += `Key Focus Areas:\n`;
      topAspects.forEach((aspect, idx) => {
        summaryText += `${idx + 1}. ${aspect.charAt(0).toUpperCase() + aspect.slice(1).replace('_', ' ')} - A primary area of discussion among users\n`;
      });
      summaryText += `\n`;
    }
    
    summaryText += `Recommendations:\n`;
    if (negativePct > 40) {
      summaryText += `• Address negative sentiment concerns proactively, particularly in ${topAspects[0] || 'key areas'}\n`;
      summaryText += `• Monitor trending topics and respond to customer pain points\n`;
    }
    summaryText += `• Leverage positive sentiment in ${topAspects[0] || 'identified areas'} to strengthen brand positioning\n`;
    summaryText += `• Continue monitoring sentiment trends to identify emerging patterns`;

    const summary = {
      summary: summaryText,
      used_llm: false, // Can be set to true if OpenAI integration is added
      stats: {
        sentiment: {
          counts: {
            positive,
            negative,
            neutral,
            total
          },
          percentages: {
            positive: parseFloat(positivePct),
            negative: parseFloat(negativePct),
            neutral: parseFloat(neutralPct)
          }
        },
        period: { start, end },
        average_sentiment_score: Number(stats.avg_sentiment) || 0,
        top_aspects: topAspects
      }
    };

    res.json(summary);
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: String(e) });
  }
});

// Structured Brief - Enhanced with data-driven structured insights
app.get("/api/structured-brief", async (req, res) => {
  try {
    if (!exists(THEMES_PARQUET)) {
      return res.status(404).json({ error: "Parquet file not found" });
    }

    const start = req.query.start;
    const end = req.query.end;
    const keyword = req.query.keyword || '';
    const whereClause = buildWhereClause(start, end);

    // Get aspect and sentiment breakdown
    const sql = `
      SELECT 
        aspect_dominant,
        sentiment_label,
        COUNT(*)::INTEGER as count
      FROM read_parquet('${THEMES_PARQUET.replace(/\\/g, "/")}')
      ${whereClause}
      GROUP BY aspect_dominant, sentiment_label
      ORDER BY count DESC;
    `;
    
    const rows = await new Promise((resolve, reject) =>
      conn.all(sql, (err, r) => (err ? reject(err) : resolve(r)))
    );

    // Get overall sentiment stats
    const sentimentSql = `
      SELECT 
        COUNT(*)::INTEGER as total,
        COUNT(CASE WHEN sentiment_label = 'positive' THEN 1 END)::INTEGER as positive,
        COUNT(CASE WHEN sentiment_label = 'negative' THEN 1 END)::INTEGER as negative,
        COUNT(CASE WHEN sentiment_label = 'neutral' THEN 1 END)::INTEGER as neutral
      FROM read_parquet('${THEMES_PARQUET.replace(/\\/g, "/")}')
      ${whereClause};
    `;
    
    const sentimentRows = await new Promise((resolve, reject) =>
      conn.all(sentimentSql, (err, r) => (err ? reject(err) : resolve(r)))
    );

    const sentimentStats = sentimentRows[0];
    const total = Number(sentimentStats.total) || 0;
    const positive = Number(sentimentStats.positive) || 0;
    const negative = Number(sentimentStats.negative) || 0;
    const neutral = Number(sentimentStats.neutral) || 0;

    // Group by aspect
    const sentimentByAspect = {};
    rows.forEach(row => {
      const aspect = normalizeAspectName(row.aspect_dominant);
      const sentiment = row.sentiment_label?.toLowerCase() || 'neutral';
      const count = Number(row.count) || 0;

      if (!sentimentByAspect[aspect]) {
        sentimentByAspect[aspect] = { positive: 0, negative: 0, neutral: 0, total: 0 };
      }
      sentimentByAspect[aspect][sentiment] = count;
      sentimentByAspect[aspect].total += count;
    });

    // Get top aspects (excluding fallback label)
    const topAspects = Object.entries(sentimentByAspect)
      .filter(([aspect]) => aspect !== ASPECT_FALLBACK_LABEL)
      .sort((a, b) => b[1].total - a[1].total)
      .slice(0, 5);

    // Generate structured insights
    const executiveBullets = [];
    executiveBullets.push(`Analyzed ${total.toLocaleString()} social media conversations from ${start} to ${end}`);
    executiveBullets.push(`Sentiment distribution: ${((positive/total)*100).toFixed(1)}% positive, ${((negative/total)*100).toFixed(1)}% negative, ${((neutral/total)*100).toFixed(1)}% neutral`);
    
    if (topAspects.length > 0) {
      const topAspect = topAspects[0];
      const aspectName = topAspect[0].charAt(0).toUpperCase() + topAspect[0].slice(1).replace('_', ' ');
      executiveBullets.push(`${aspectName} is the most discussed topic with ${topAspect[1].total.toLocaleString()} mentions`);
    }
    
    if (negative > positive * 1.5) {
      executiveBullets.push(`Negative sentiment significantly outweighs positive, indicating areas requiring immediate attention`);
    }

    // Generate themes
    const themes = topAspects
      .slice(0, 4)
      .map(([aspect]) => aspect.charAt(0).toUpperCase() + aspect.slice(1).replace('_', ' '));

    // Generate risks based on negative sentiment
    const risks = [];
    topAspects.forEach(([aspect, data]) => {
      const negativePct = data.total > 0 ? (data.negative / data.total) * 100 : 0;
      if (negativePct > 50) {
        const aspectName = aspect.charAt(0).toUpperCase() + aspect.slice(1).replace('_', ' ');
        risks.push(`High negative sentiment (${negativePct.toFixed(1)}%) in ${aspectName} indicates customer dissatisfaction requiring urgent intervention`);
      }
    });
    
    if (negative > total * 0.4) {
      risks.push(`Overall negative sentiment exceeds 40%, suggesting systemic issues that need comprehensive review`);
    }

    // Generate opportunities based on positive sentiment
    const opportunities = [];
    topAspects.forEach(([aspect, data]) => {
      const positivePct = data.total > 0 ? (data.positive / data.total) * 100 : 0;
      if (positivePct > 30) {
        const aspectName = aspect.charAt(0).toUpperCase() + aspect.slice(1).replace('_', ' ');
        opportunities.push(`Strong positive sentiment (${positivePct.toFixed(1)}%) in ${aspectName} presents an opportunity to amplify and leverage customer satisfaction`);
      }
    });
    
    if (positive > total * 0.25) {
      opportunities.push(`Significant positive sentiment base (${((positive/total)*100).toFixed(1)}%) can be leveraged for brand advocacy and marketing initiatives`);
    }
    
    if (opportunities.length === 0) {
      opportunities.push(`Focus on improving customer experience to increase positive sentiment across all aspects`);
    }

    // Generate executive text
    const executiveText = `Executive Brief: ${total.toLocaleString()} conversations analyzed from ${start} to ${end}. `;
    const executiveText2 = `Sentiment analysis reveals ${((negative/total)*100).toFixed(1)}% negative, ${((positive/total)*100).toFixed(1)}% positive, and ${((neutral/total)*100).toFixed(1)}% neutral sentiment. `;
    const executiveText3 = `Key focus areas include ${themes.slice(0, 3).join(', ')}. `;
    const executiveText4 = risks.length > 0 ? `Primary risks: ${risks[0]}. ` : '';
    const executiveText5 = opportunities.length > 0 ? `Key opportunities: ${opportunities[0]}.` : '';

    const brief = {
      executive_text: executiveText + executiveText2 + executiveText3 + executiveText4 + executiveText5,
      structured: {
        executive_bullets: executiveBullets,
        themes: themes,
        risks: risks.length > 0 ? risks : [`Monitor sentiment trends closely for emerging issues`],
        opportunities: opportunities
      },
      period: { start, end },
      total_tweets: total,
      sentiment_by_aspect: Object.fromEntries(
        Object.entries(sentimentByAspect)
          .filter(([aspect]) => aspect !== ASPECT_FALLBACK_LABEL)
          .map(([aspect, data]) => [aspect, {
            positive: data.positive,
            negative: data.negative,
            neutral: data.neutral,
            total: data.total
          }])
      )
    };

    res.json(brief);
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: String(e) });
  }
});

// Raw Tweets Download
app.get("/api/tweets/raw", async (req, res) => {
  try {
    if (!exists(THEMES_PARQUET)) {
      return res.status(404).json({ error: "Parquet file not found" });
    }

    const start = req.query.start;
    const end = req.query.end;
    const format = req.query.format || 'csv';
    const whereClause = buildWhereClause(start, end);

    const sql = `
      SELECT 
        id, twitterurl, text_clean, text, fulltext,
        sentiment_label, sentiment_score, 
        aspect_dominant,
        createdat
      FROM read_parquet('${THEMES_PARQUET.replace(/\\/g, "/")}')
      ${whereClause}
      ORDER BY createdat DESC;
    `;
    
    const rows = await new Promise((resolve, reject) =>
      conn.all(sql, (err, r) => (err ? reject(err) : resolve(r)))
    );

    const serializedRows = serializeRows(rows);

    if (format === 'csv') {
      // Generate CSV content
      const headers = ['ID', 'Twitter URL', 'Text Clean', 'Text', 'Full Text', 'Sentiment Label', 'Sentiment Score', 'Aspect Dominant', 'Created At'];
      const csvRows = serializedRows.map(row => [
        row.id || '',
        row.twitterurl || '',
        (row.text_clean || '').replace(/"/g, '""'),
        (row.text || '').replace(/"/g, '""'),
        (row.fulltext || '').replace(/"/g, '""'),
        row.sentiment_label || '',
        row.sentiment_score || '',
        row.aspect_dominant || '',
        row.createdat || ''
      ]);

      const csvContent = [
        headers.map(h => `"${h}"`).join(','),
        ...csvRows.map(row => row.map(cell => `"${String(cell)}"`).join(','))
      ].join('\n');

      res.setHeader('Content-Type', 'text/csv');
      res.setHeader('Content-Disposition', `attachment; filename="raw_tweets_${start}_to_${end}.csv"`);
      res.send(csvContent);
    } else if (format === 'xlsx') {
      // Generate proper Excel file using ExcelJS
      const workbook = new ExcelJS.Workbook();
      const worksheet = workbook.addWorksheet('Raw Tweets');
      
      // Add headers
      worksheet.columns = [
        { header: 'ID', key: 'id', width: 20 },
        { header: 'Twitter URL', key: 'twitterurl', width: 50 },
        { header: 'Text Clean', key: 'text_clean', width: 50 },
        { header: 'Text', key: 'text', width: 50 },
        { header: 'Full Text', key: 'fulltext', width: 50 },
        { header: 'Sentiment Label', key: 'sentiment_label', width: 15 },
        { header: 'Sentiment Score', key: 'sentiment_score', width: 15 },
        { header: 'Aspect Dominant', key: 'aspect_dominant', width: 20 },
        { header: 'Created At', key: 'createdat', width: 15 }
      ];
      
      // Style header row
      worksheet.getRow(1).font = { bold: true };
      worksheet.getRow(1).fill = {
        type: 'pattern',
        pattern: 'solid',
        fgColor: { argb: 'FFE0E0E0' }
      };
      
      // Add data rows
      serializedRows.forEach(row => {
        worksheet.addRow({
          id: row.id || '',
          twitterurl: row.twitterurl || '',
          text_clean: row.text_clean || '',
          text: row.text || '',
          fulltext: row.fulltext || '',
          sentiment_label: row.sentiment_label || '',
          sentiment_score: row.sentiment_score || '',
          aspect_dominant: row.aspect_dominant || '',
          createdat: row.createdat || ''
        });
      });
      
      // Set response headers
      res.setHeader('Content-Type', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet');
      res.setHeader('Content-Disposition', `attachment; filename="raw_tweets_${start}_to_${end}.xlsx"`);
      
      // Write to response
      await workbook.xlsx.write(res);
      res.end();
    } else {
      res.status(400).json({ error: 'Unsupported format. Use csv or xlsx' });
    }
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: String(e) });
  }
});

// Sentiment Report Download
app.get("/api/reports/sentiment", async (req, res) => {
  try {
    if (!exists(THEMES_PARQUET)) {
      return res.status(404).json({ error: "Parquet file not found" });
    }

    const start = req.query.start;
    const end = req.query.end;
    const format = req.query.format || 'pdf';
    const whereClause = buildWhereClause(start, end);

    // Get sentiment summary
    const summarySql = `
      SELECT 
        sentiment_label,
        COUNT(*)::INTEGER as count,
        AVG(sentiment_score)::DOUBLE as avg_score
      FROM read_parquet('${THEMES_PARQUET.replace(/\\/g, "/")}')
      ${whereClause}
      GROUP BY sentiment_label;
    `;
    
    const summaryRows = await new Promise((resolve, reject) =>
      conn.all(summarySql, (err, r) => (err ? reject(err) : resolve(r)))
    );

    // Get daily trend
    // The createdat column is a timestamp, so we can use DATE() directly
    const trendSql = `
      SELECT 
        DATE(createdat) as date,
        sentiment_label,
        COUNT(*)::INTEGER as count
      FROM read_parquet('${THEMES_PARQUET.replace(/\\/g, "/")}')
      ${whereClause}
      GROUP BY DATE(createdat), sentiment_label
      ORDER BY date DESC;
    `;
    
    const trendRows = await new Promise((resolve, reject) =>
      conn.all(trendSql, (err, r) => (err ? reject(err) : resolve(r)))
    );

    const summary = { positive: 0, negative: 0, neutral: 0, total: 0 };
    summaryRows.forEach(row => {
      const label = row.sentiment_label?.toLowerCase() || 'neutral';
      summary[label] = Number(row.count) || 0;
      summary.total += summary[label];
    });

    if (format === 'pdf') {
      // Generate PDF using PDFKit
      const doc = new PDFDocument({ margin: 50 });
      res.setHeader('Content-Type', 'application/pdf');
      res.setHeader('Content-Disposition', `attachment; filename="sentiment_report_${start}_to_${end}.pdf"`);
      doc.pipe(res);
      
      // Add title
      doc.fontSize(20).text('Sentiment Analysis Report', { align: 'center' });
      doc.moveDown();
      doc.fontSize(12).text(`Date Range: ${start} to ${end}`, { align: 'center' });
      doc.text(`Generated: ${new Date().toISOString().split('T')[0]}`, { align: 'center' });
      doc.moveDown(2);
      
      // Add summary
      doc.fontSize(16).text('Summary', { underline: true });
      doc.moveDown();
      doc.fontSize(12);
      doc.text(`Total Tweets: ${summary.total.toLocaleString()}`);
      doc.text(`Positive: ${summary.positive} (${((summary.positive/summary.total)*100).toFixed(1)}%)`);
      doc.text(`Negative: ${summary.negative} (${((summary.negative/summary.total)*100).toFixed(1)}%)`);
      doc.text(`Neutral: ${summary.neutral} (${((summary.neutral/summary.total)*100).toFixed(1)}%)`);
      doc.moveDown(2);
      
      // Add daily trend table
      doc.fontSize(16).text('Daily Trend', { underline: true });
      doc.moveDown();
      
      const trendData = Object.entries(
        trendRows.reduce((acc, row) => {
          const date = row.date instanceof Date ? row.date.toISOString().split('T')[0] : String(row.date).split('T')[0];
          if (!acc[date]) acc[date] = { positive: 0, negative: 0, neutral: 0 };
          const label = row.sentiment_label?.toLowerCase() || 'neutral';
          acc[date][label] = Number(row.count) || 0;
          return acc;
        }, {})
      );
      
      // Table headers
      const startX = 50;
      let yPos = doc.y;
      const pageHeight = 792; // Standard page height (A4: 842, but accounting for margins)
      const rowHeight = 20;
      
      doc.fontSize(10).font('Helvetica-Bold');
      doc.text('Date', startX, yPos);
      doc.text('Positive', startX + 100, yPos);
      doc.text('Negative', startX + 180, yPos);
      doc.text('Neutral', startX + 260, yPos);
      doc.text('Total', startX + 340, yPos);
      
      // Table rows with pagination
      doc.font('Helvetica');
      trendData.forEach(([date, data]) => {
        // Check if we need a new page (leave some margin at bottom)
        if (yPos + rowHeight > pageHeight - 50) {
          doc.addPage();
          yPos = 50; // Reset to top of new page
          
          // Re-draw headers on new page
          doc.fontSize(10).font('Helvetica-Bold');
          doc.text('Date', startX, yPos);
          doc.text('Positive', startX + 100, yPos);
          doc.text('Negative', startX + 180, yPos);
          doc.text('Neutral', startX + 260, yPos);
          doc.text('Total', startX + 340, yPos);
          doc.font('Helvetica');
          yPos += rowHeight; // Move past header row
        }
        
        // Draw the data row
        yPos += rowHeight;
        doc.text(date, startX, yPos);
        doc.text(String(data.positive), startX + 100, yPos);
        doc.text(String(data.negative), startX + 180, yPos);
        doc.text(String(data.neutral), startX + 260, yPos);
        doc.text(String(data.positive + data.negative + data.neutral), startX + 340, yPos);
      });
      
      doc.end();
    } else if (format === 'xlsx') {
      // Generate proper Excel file
      const workbook = new ExcelJS.Workbook();
      const summarySheet = workbook.addWorksheet('Summary');
      summarySheet.columns = [
        { header: 'Metric', key: 'metric', width: 20 },
        { header: 'Value', key: 'value', width: 15 }
      ];
      summarySheet.getRow(1).font = { bold: true };
      summarySheet.getRow(1).fill = {
        type: 'pattern',
        pattern: 'solid',
        fgColor: { argb: 'FFE0E0E0' }
      };
      summarySheet.addRow({ metric: 'Total Tweets', value: summary.total });
      summarySheet.addRow({ metric: 'Positive', value: `${summary.positive} (${((summary.positive/summary.total)*100).toFixed(1)}%)` });
      summarySheet.addRow({ metric: 'Negative', value: `${summary.negative} (${((summary.negative/summary.total)*100).toFixed(1)}%)` });
      summarySheet.addRow({ metric: 'Neutral', value: `${summary.neutral} (${((summary.neutral/summary.total)*100).toFixed(1)}%)` });
      
      const worksheet = workbook.addWorksheet('Daily Trend');
      worksheet.columns = [
        { header: 'Date', key: 'date', width: 12 },
        { header: 'Positive', key: 'positive', width: 12 },
        { header: 'Negative', key: 'negative', width: 12 },
        { header: 'Neutral', key: 'neutral', width: 12 },
        { header: 'Total', key: 'total', width: 12 }
      ];
      worksheet.getRow(1).font = { bold: true };
      worksheet.getRow(1).fill = {
        type: 'pattern',
        pattern: 'solid',
        fgColor: { argb: 'FFE0E0E0' }
      };
      
      const trendData = Object.entries(
        trendRows.reduce((acc, row) => {
          const date = row.date instanceof Date ? row.date.toISOString().split('T')[0] : String(row.date).split('T')[0];
          if (!acc[date]) acc[date] = { positive: 0, negative: 0, neutral: 0 };
          const label = row.sentiment_label?.toLowerCase() || 'neutral';
          acc[date][label] = Number(row.count) || 0;
          return acc;
        }, {})
      );
      
      trendData.forEach(([date, data]) => {
        worksheet.addRow({
          date: date,
          positive: data.positive,
          negative: data.negative,
          neutral: data.neutral,
          total: data.positive + data.negative + data.neutral
        });
      });
      
      res.setHeader('Content-Type', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet');
      res.setHeader('Content-Disposition', `attachment; filename="sentiment_report_${start}_to_${end}.xlsx"`);
      await workbook.xlsx.write(res);
      res.end();
    } else {
      res.status(400).json({ error: 'Unsupported format. Use pdf or xlsx' });
    }
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: String(e) });
  }
});

// Aspect Report Download
app.get("/api/reports/aspects", async (req, res) => {
  try {
    if (!exists(THEMES_PARQUET)) {
      return res.status(404).json({ error: "Parquet file not found" });
    }

    const start = req.query.start;
    const end = req.query.end;
    const format = req.query.format || 'pdf';
    const whereClause = buildWhereClause(start, end);

    const sql = `
      SELECT 
        aspect_dominant,
        sentiment_label,
        COUNT(*)::INTEGER as count
      FROM read_parquet('${THEMES_PARQUET.replace(/\\/g, "/")}')
      ${whereClause}
      GROUP BY aspect_dominant, sentiment_label
      ORDER BY aspect_dominant, sentiment_label;
    `;
    
    const rows = await new Promise((resolve, reject) =>
      conn.all(sql, (err, r) => (err ? reject(err) : resolve(r)))
    );

    // Group by aspect
    const aspectMap = {};
    rows.forEach(row => {
      const aspect = normalizeAspectName(row.aspect_dominant);
      const sentiment = row.sentiment_label?.toLowerCase() || 'neutral';
      const count = Number(row.count) || 0;

      if (!aspectMap[aspect]) {
        aspectMap[aspect] = { positive: 0, negative: 0, neutral: 0, total: 0 };
      }
      aspectMap[aspect][sentiment] = count;
      aspectMap[aspect].total += count;
    });

    const aspects = Object.entries(aspectMap)
      .filter(([aspect]) => aspect !== ASPECT_FALLBACK_LABEL)
      .sort((a, b) => b[1].total - a[1].total);

    if (format === 'pdf') {
      // Generate PDF using PDFKit
      const doc = new PDFDocument({ margin: 50 });
      res.setHeader('Content-Type', 'application/pdf');
      res.setHeader('Content-Disposition', `attachment; filename="aspect_report_${start}_to_${end}.pdf"`);
      doc.pipe(res);
      
      doc.fontSize(20).text('Aspect Analysis Report', { align: 'center' });
      doc.moveDown();
      doc.fontSize(12).text(`Date Range: ${start} to ${end}`, { align: 'center' });
      doc.text(`Generated: ${new Date().toISOString().split('T')[0]}`, { align: 'center' });
      doc.moveDown(2);
      
      doc.fontSize(16).text('Aspect × Sentiment Breakdown', { underline: true });
      doc.moveDown();
      
      const startX = 50;
      let yPos = doc.y;
      doc.fontSize(10).font('Helvetica-Bold');
      doc.text('Aspect', startX, yPos);
      doc.text('Positive', startX + 100, yPos);
      doc.text('Negative', startX + 180, yPos);
      doc.text('Neutral', startX + 260, yPos);
      doc.text('Total', startX + 340, yPos);
      doc.text('Pos %', startX + 400, yPos);
      doc.text('Neg %', startX + 450, yPos);
      
      doc.font('Helvetica');
      aspects.forEach(([aspect, data]) => {
        if (yPos > 700) {
          doc.addPage();
          yPos = 50;
        }
        yPos += 20;
        const aspectName = aspect.charAt(0).toUpperCase() + aspect.slice(1).replace('_', ' ');
        doc.text(aspectName, startX, yPos);
        doc.text(String(data.positive), startX + 100, yPos);
        doc.text(String(data.negative), startX + 180, yPos);
        doc.text(String(data.neutral), startX + 260, yPos);
        doc.text(String(data.total), startX + 340, yPos);
        doc.text(`${data.total > 0 ? ((data.positive / data.total) * 100).toFixed(1) : 0}%`, startX + 400, yPos);
        doc.text(`${data.total > 0 ? ((data.negative / data.total) * 100).toFixed(1) : 0}%`, startX + 450, yPos);
      });
      
      doc.end();
    } else if (format === 'xlsx') {
      // Generate proper Excel file
      const workbook = new ExcelJS.Workbook();
      const worksheet = workbook.addWorksheet('Aspect Analysis');
      
      worksheet.columns = [
        { header: 'Aspect', key: 'aspect', width: 20 },
        { header: 'Positive', key: 'positive', width: 12 },
        { header: 'Negative', key: 'negative', width: 12 },
        { header: 'Neutral', key: 'neutral', width: 12 },
        { header: 'Total', key: 'total', width: 12 },
        { header: 'Positive %', key: 'positive_pct', width: 12 },
        { header: 'Negative %', key: 'negative_pct', width: 12 },
        { header: 'Neutral %', key: 'neutral_pct', width: 12 }
      ];
      
      worksheet.getRow(1).font = { bold: true };
      worksheet.getRow(1).fill = {
        type: 'pattern',
        pattern: 'solid',
        fgColor: { argb: 'FFE0E0E0' }
      };
      
      aspects.forEach(([aspect, data]) => {
        worksheet.addRow({
          aspect: aspect.charAt(0).toUpperCase() + aspect.slice(1).replace('_', ' '),
          positive: data.positive,
          negative: data.negative,
          neutral: data.neutral,
          total: data.total,
          positive_pct: data.total > 0 ? ((data.positive / data.total) * 100).toFixed(1) : 0,
          negative_pct: data.total > 0 ? ((data.negative / data.total) * 100).toFixed(1) : 0,
          neutral_pct: data.total > 0 ? ((data.neutral / data.total) * 100).toFixed(1) : 0
        });
      });
      
      res.setHeader('Content-Type', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet');
      res.setHeader('Content-Disposition', `attachment; filename="aspect_report_${start}_to_${end}.xlsx"`);
      await workbook.xlsx.write(res);
      res.end();
    } else {
      res.status(400).json({ error: 'Unsupported format. Use pdf or xlsx' });
    }
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: String(e) });
  }
});

// Theme Report Download
app.get("/api/reports/themes", async (req, res) => {
  try {
    if (!exists(THEMES_PARQUET)) {
      return res.status(404).json({ error: "Parquet file not found" });
    }

    const start = req.query.start;
    const end = req.query.end;
    const format = req.query.format || 'pdf';
    const whereClause = buildWhereClause(start, end);

    // Get themes directly from parquet - use same query as /api/themes
    const themesSql = `
      SELECT theme::INTEGER AS id, COUNT(*)::INTEGER AS tweet_count
      FROM read_parquet('${THEMES_PARQUET.replace(/\\/g, "/")}')
      ${whereClause}
      GROUP BY theme
      ORDER BY tweet_count DESC;
    `;
    
    const themeRows = await new Promise((resolve, reject) =>
      conn.all(themesSql, (err, r) => (err ? reject(err) : resolve(r)))
    );

    // Get theme names and summaries if available
    const names = exists(THEME_NAMES) ? JSON.parse(fs.readFileSync(THEME_NAMES, "utf8")) : {};
    const sums = exists(THEME_SUMMARIES) ? JSON.parse(fs.readFileSync(THEME_SUMMARIES, "utf8")) : {};
    
    const themes = themeRows.map(row => ({
      id: row.id || 0,
      name: names[row.id] || `Theme ${row.id || 0}`,
      tweet_count: Number(row.tweet_count) || 0,
      summary: sums[row.id] || `Analysis of ${row.tweet_count} tweets in this theme`
    }));

    if (format === 'pdf') {
      // Generate PDF using PDFKit
      const doc = new PDFDocument({ margin: 50 });
      res.setHeader('Content-Type', 'application/pdf');
      res.setHeader('Content-Disposition', `attachment; filename="theme_report_${start}_to_${end}.pdf"`);
      doc.pipe(res);
      
      doc.fontSize(20).text('Theme Analysis Report', { align: 'center' });
      doc.moveDown();
      doc.fontSize(12).text(`Date Range: ${start} to ${end}`, { align: 'center' });
      doc.text(`Generated: ${new Date().toISOString().split('T')[0]}`, { align: 'center' });
      doc.moveDown(2);
      
      doc.fontSize(16).text('Identified Themes', { underline: true });
      doc.moveDown();
      
      themes.forEach((theme, index) => {
        if (doc.y > 700) {
          doc.addPage();
        }
        doc.fontSize(14).font('Helvetica-Bold').text(`Theme ${theme.id}: ${theme.name}`);
        doc.moveDown(0.5);
        doc.fontSize(10).font('Helvetica').text(`Tweet Count: ${theme.tweet_count.toLocaleString()}`);
        doc.moveDown(0.5);
        doc.text(theme.summary, { width: 500 });
        doc.moveDown(1.5);
      });
      
      doc.end();
    } else if (format === 'xlsx') {
      // Generate proper Excel file
      const workbook = new ExcelJS.Workbook();
      const worksheet = workbook.addWorksheet('Themes');
      
      worksheet.columns = [
        { header: 'Theme ID', key: 'id', width: 12 },
        { header: 'Theme Name', key: 'name', width: 40 },
        { header: 'Tweet Count', key: 'tweet_count', width: 15 },
        { header: 'Summary', key: 'summary', width: 80 }
      ];
      
      worksheet.getRow(1).font = { bold: true };
      worksheet.getRow(1).fill = {
        type: 'pattern',
        pattern: 'solid',
        fgColor: { argb: 'FFE0E0E0' }
      };
      
      // Enable word wrap for summary column
      worksheet.getColumn(4).alignment = { wrapText: true };
      
      themes.forEach(theme => {
        worksheet.addRow({
          id: theme.id,
          name: theme.name,
          tweet_count: theme.tweet_count,
          summary: theme.summary
        });
      });
      
      res.setHeader('Content-Type', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet');
      res.setHeader('Content-Disposition', `attachment; filename="theme_report_${start}_to_${end}.xlsx"`);
      await workbook.xlsx.write(res);
      res.end();
    } else {
      res.status(400).json({ error: 'Unsupported format. Use pdf or xlsx' });
    }
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: String(e) });
  }
});

// Analytics Dashboard Report Download
app.get("/api/reports/dashboard", async (req, res) => {
  try {
    if (!exists(THEMES_PARQUET)) {
      return res.status(404).json({ error: "Parquet file not found" });
    }

    const start = req.query.start || null;
    const end = req.query.end || null;
    const format = (req.query.format || 'pdf').toLowerCase();
    const whereClause = buildWhereClause(start, end);

    const summarySql = `
      SELECT 
        sentiment_label,
        COUNT(*)::INTEGER as count
      FROM read_parquet('${THEMES_PARQUET.replace(/\\/g, "/")}')
      ${whereClause}
      GROUP BY sentiment_label;
    `;

    const trendSql = `
      SELECT 
        DATE(createdat) as date,
        sentiment_label,
        COUNT(*)::INTEGER as count
      FROM read_parquet('${THEMES_PARQUET.replace(/\\/g, "/")}')
      ${whereClause}
      GROUP BY DATE(createdat), sentiment_label
      ORDER BY date DESC;
    `;

    const aspectSql = `
      SELECT 
        aspect_dominant,
        sentiment_label,
        COUNT(*)::INTEGER as count
      FROM read_parquet('${THEMES_PARQUET.replace(/\\/g, "/")}')
      ${whereClause}
      GROUP BY aspect_dominant, sentiment_label
      ORDER BY aspect_dominant, sentiment_label;
    `;

    const [summaryRows, trendRows, aspectRows] = await Promise.all([
      new Promise((resolve, reject) => conn.all(summarySql, (err, r) => (err ? reject(err) : resolve(r)))),
      new Promise((resolve, reject) => conn.all(trendSql, (err, r) => (err ? reject(err) : resolve(r)))),
      new Promise((resolve, reject) => conn.all(aspectSql, (err, r) => (err ? reject(err) : resolve(r))))
    ]);

    const summary = { positive: 0, negative: 0, neutral: 0, total: 0 };
    summaryRows.forEach(row => {
      const label = row.sentiment_label?.toLowerCase() || 'neutral';
      const count = Number(row.count) || 0;
      if (summary[label] !== undefined) {
        summary[label] += count;
      }
      summary.total += count;
    });

    const trendData = Object.entries(
      trendRows.reduce((acc, row) => {
        const dateKey = row.date instanceof Date ? row.date.toISOString().split('T')[0] : String(row.date).split('T')[0];
        if (!acc[dateKey]) {
          acc[dateKey] = { positive: 0, negative: 0, neutral: 0 };
        }
        const label = row.sentiment_label?.toLowerCase() || 'neutral';
        acc[dateKey][label] = Number(row.count) || 0;
        return acc;
      }, {})
    ).sort((a, b) => new Date(a[0]) - new Date(b[0]));

    const aspectMap = aspectRows.reduce((acc, row) => {
      const aspect = normalizeAspectName(row.aspect_dominant);
      const label = row.sentiment_label?.toLowerCase() || 'neutral';
      const count = Number(row.count) || 0;
      if (!acc[aspect]) {
        acc[aspect] = { positive: 0, negative: 0, neutral: 0, total: 0 };
      }
      acc[aspect][label] += count;
      acc[aspect].total += count;
      return acc;
    }, {});

    const aspectData = Object.entries(aspectMap)
      .filter(([aspect]) => aspect !== ASPECT_FALLBACK_LABEL)
      .sort((a, b) => b[1].total - a[1].total)
      .slice(0, 12);

    if (format !== 'pdf') {
      return res.status(400).json({ error: 'Unsupported format. Only PDF exports are available for the dashboard report.' });
    }

    const doc = new PDFDocument({ margin: 50 });
    res.setHeader('Content-Type', 'application/pdf');
    const startLabel = start || 'all';
    const endLabel = end || 'all';
    res.setHeader('Content-Disposition', `attachment; filename="analytics_dashboard_${startLabel}_to_${endLabel}.pdf"`);
    doc.pipe(res);

    const prettyDateRange = start && end ? `${start} to ${end}` : 'All Available Data';

    doc.fontSize(20).text('Analytics Dashboard Report', { align: 'center' });
    doc.moveDown();
    doc.fontSize(12).text(`Date Range: ${prettyDateRange}`, { align: 'center' });
    doc.text(`Generated: ${new Date().toISOString().split('T')[0]}`, { align: 'center' });
    doc.moveDown(2);

    doc.fontSize(16).text('Sentiment Summary', { underline: true });
    doc.moveDown();
    doc.fontSize(12);

    const pct = (value) => {
      if (!summary.total) return '0.0';
      return ((value / summary.total) * 100).toFixed(1);
    };

    doc.text(`Total Tweets: ${summary.total.toLocaleString()}`);
    doc.text(`Positive: ${summary.positive.toLocaleString()} (${pct(summary.positive)}%)`);
    doc.text(`Negative: ${summary.negative.toLocaleString()} (${pct(summary.negative)}%)`);
    doc.text(`Neutral: ${summary.neutral.toLocaleString()} (${pct(summary.neutral)}%)`);
    doc.moveDown(1.5);

    doc.fontSize(16).text('Sentiment Trend', { underline: true });
    doc.moveDown();

    const startX = 50;
    let yPos = doc.y;
    const rowHeight = 18;
    const pageHeight = doc.page.height - doc.page.margins.bottom;

    doc.fontSize(10).font('Helvetica-Bold');
    doc.text('Date', startX, yPos);
    doc.text('Positive', startX + 90, yPos);
    doc.text('Negative', startX + 160, yPos);
    doc.text('Neutral', startX + 230, yPos);
    doc.text('Total', startX + 300, yPos);
    doc.font('Helvetica');

    yPos += rowHeight;

    trendData.slice(-60).forEach(([date, counts]) => {
      if (yPos + rowHeight > pageHeight) {
        doc.addPage();
        yPos = doc.page.margins.top;
        doc.fontSize(10).font('Helvetica-Bold');
        doc.text('Date', startX, yPos);
        doc.text('Positive', startX + 90, yPos);
        doc.text('Negative', startX + 160, yPos);
        doc.text('Neutral', startX + 230, yPos);
        doc.text('Total', startX + 300, yPos);
        doc.font('Helvetica');
        yPos += rowHeight;
      }

      const totalCount = (counts.positive || 0) + (counts.negative || 0) + (counts.neutral || 0);
      doc.text(String(date), startX, yPos);
      doc.text(String(counts.positive || 0), startX + 90, yPos);
      doc.text(String(counts.negative || 0), startX + 160, yPos);
      doc.text(String(counts.neutral || 0), startX + 230, yPos);
      doc.text(String(totalCount), startX + 300, yPos);
      yPos += rowHeight;
    });

    doc.addPage();
    doc.fontSize(16).text('Top Aspect Drivers', { underline: true });
    doc.moveDown();

    doc.fontSize(10).font('Helvetica-Bold');
    doc.text('Aspect', startX, doc.y);
    doc.text('Positive', startX + 120, doc.y);
    doc.text('Negative', startX + 190, doc.y);
    doc.text('Neutral', startX + 260, doc.y);
    doc.text('Total', startX + 330, doc.y);
    doc.text('Positive %', startX + 390, doc.y);
    doc.text('Negative %', startX + 460, doc.y);
    doc.font('Helvetica');

    yPos = doc.y + rowHeight;
    aspectData.forEach(([aspect, counts]) => {
      if (yPos + rowHeight > pageHeight) {
        doc.addPage();
        yPos = doc.page.margins.top;
        doc.fontSize(10).font('Helvetica-Bold');
        doc.text('Aspect', startX, yPos);
        doc.text('Positive', startX + 120, yPos);
        doc.text('Negative', startX + 190, yPos);
        doc.text('Neutral', startX + 260, yPos);
        doc.text('Total', startX + 330, yPos);
        doc.text('Positive %', startX + 390, yPos);
        doc.text('Negative %', startX + 460, yPos);
        doc.font('Helvetica');
        yPos += rowHeight;
      }

      const posPct = counts.total ? ((counts.positive / counts.total) * 100).toFixed(1) : '0.0';
      const negPct = counts.total ? ((counts.negative / counts.total) * 100).toFixed(1) : '0.0';

      const aspectName = aspect.charAt(0).toUpperCase() + aspect.slice(1).replace(/_/g, ' ');
      doc.text(aspectName, startX, yPos);
      doc.text(String(counts.positive), startX + 120, yPos);
      doc.text(String(counts.negative), startX + 190, yPos);
      doc.text(String(counts.neutral), startX + 260, yPos);
      doc.text(String(counts.total), startX + 330, yPos);
      doc.text(`${posPct}%`, startX + 390, yPos);
      doc.text(`${negPct}%`, startX + 460, yPos);
      yPos += rowHeight;
    });

    doc.end();
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: String(e) });
  }
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => console.log(`API running on http://localhost:${PORT}`));
