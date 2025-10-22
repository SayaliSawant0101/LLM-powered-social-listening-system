import express from "express";
import cors from "cors";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import duckdb from "duckdb";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// 👇 points to the existing root-level data folder
const DATA_DIR = path.join(__dirname, "../data");

// optional static payload (if you put a copy here)
const THEMES_PAYLOAD = path.join(__dirname, "../frontend/public/themes_payload.json");

// parquet + names/summaries in root/data
const THEMES_PARQUET = path.join(DATA_DIR, "tweets_stage3_themes.parquet");
const THEME_NAMES = path.join(DATA_DIR, "theme_names.json");
const THEME_SUMMARIES = path.join(DATA_DIR, "theme_summaries.json");

const app = express();
app.use(cors());
app.use(express.json());

const db = new duckdb.Database(":memory:");
const conn = db.connect();

const exists = (p) => { try { fs.accessSync(p); return true; } catch { return false; } };

// List themes (uses static JSON if present; else computes from parquet)
app.get("/api/themes", async (req, res) => {
  try {
    if (exists(THEMES_PAYLOAD)) {
      const payload = JSON.parse(fs.readFileSync(THEMES_PAYLOAD, "utf8"));
      return res.json(payload);
    }

    const names = exists(THEME_NAMES) ? JSON.parse(fs.readFileSync(THEME_NAMES, "utf8")) : {};
    const sums  = exists(THEME_SUMMARIES) ? JSON.parse(fs.readFileSync(THEME_SUMMARIES, "utf8")) : {};

    const sql = `
      SELECT theme::INTEGER AS id, COUNT(*)::INTEGER AS tweet_count
      FROM read_parquet('${THEMES_PARQUET.replace(/\\/g, "/")}')
      GROUP BY theme
      ORDER BY tweet_count DESC;
    `;
    const rows = await new Promise((resolve, reject) =>
      conn.all(sql, (err, r) => (err ? reject(err) : resolve(r)))
    );

    const themes = rows.map(r => ({
      id: r.id,
      name: names[String(r.id)] ?? `Theme ${r.id}`,
      summary: sums[String(r.id)] ?? "",
      tweet_count: r.tweet_count
    }));

    res.json({ themes, updated_at: new Date().toISOString() });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: String(e) });
  }
});

// Drill-down tweets for a theme
app.get("/api/themes/:id/tweets", async (req, res) => {
  try {
    const themeId = Number(req.params.id);
    const limit = Math.min(Number(req.query.limit || 20), 200);
    const q = (req.query.q || "").toString().trim();

    const defaultFields = [
      "id", "twitterurl",
      "text_clean", "text", "clean_tweet",
      "sentiment_label", "sentiment_score",
      "aspect_pricing", "aspect_delivery", "aspect_returns", "aspect_staff",
      `"aspect_app/ux" AS aspect_app_ux`,
      "aspect_dominant",
      "date", "createdat", "lang", "has_url", "has_hashtag"
    ];
    const fields = (req.query.fields ? req.query.fields.split(",") : defaultFields).join(", ");

    const where = [`theme = ${themeId}`];
    if (q) {
      const like = `%${q.replace(/'/g, "''")}%`;
      where.push(`(text_clean ILIKE '${like}' OR text ILIKE '${like}' OR clean_tweet ILIKE '${like}')`);
    }

    const sql = `
      SELECT ${fields}
      FROM read_parquet('${THEMES_PARQUET.replace(/\\/g, "/")}')
      WHERE ${where.join(" AND ")}
      ORDER BY COALESCE(createdat, date) DESC
      LIMIT ${limit};
    `;
    const rows = await new Promise((resolve, reject) =>
      conn.all(sql, (err, r) => (err ? reject(err) : resolve(r)))
    );

    res.json({ theme: themeId, count: rows.length, items: rows });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: String(e) });
  }
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => console.log(`API running on http://localhost:${PORT}`));
