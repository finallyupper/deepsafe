import fs from "fs";
import path from "path";

export default function handler(req, res) {
  const baseDir = path.join(process.cwd(), "public");
  const categories = ["chu", "cha", "byeon", "winter"];
  let images = {};

  categories.forEach((category) => {
    const categoryPath = path.join(baseDir, category);
    try {
      const files = fs.readdirSync(categoryPath);
      images[category] = files.map((file) => `/${category}/${file}`);
    } catch (error) {
      console.error(`Error reading ${categoryPath}:`, error);
      images[category] = [];
    }
  });

  res.status(200).json(images);
}
