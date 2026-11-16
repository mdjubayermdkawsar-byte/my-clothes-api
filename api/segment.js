import { pipeline, env } from '@xenova/transformers';
import sharp from 'sharp';

// Vercel Serverless-এ রাইট পারমিশন শুধু /tmp ফোল্ডারে থাকে
env.cacheDir = '/tmp/transformers_cache';
env.allowLocalModels = false;

let segmenter = null;

async function getSegmenter() {
  if (!segmenter) {
    segmenter = await pipeline('image-segmentation', 'Xenova/segformer_b2_clothes', {
      quantized: true,
    });
  }
  return segmenter;
}

const LABEL_MAP = {
  'Background': 'Background',
  'Hat': 'Hat',
  'Hair': 'Hair',
  'Sunglasses': 'Glasses',
  'Upper-clothes': 'Shirt/Top',
  'Skirt': 'Pants/Skirt',
  'Pants': 'Pants/Skirt',
  'Dress': 'Shirt/Top',
  'Belt': 'Belt',
  'Left-shoe': 'Shoes',
  'Right-shoe': 'Shoes',
  'Face': 'Face',
  'Left-leg': 'Skin/Body',
  'Right-leg': 'Skin/Body',
  'Left-arm': 'Skin/Body',
  'Right-arm': 'Skin/Body',
  'Bag': 'Bag',
  'Scarf': 'Shirt/Top'
};

export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ success: false, error: 'Only POST method allowed' });
  }

  try {
    const { image } = req.body || {};
    if (!image) {
      return res.status(400).json({ success: false, error: 'image base64 is required' });
    }

    const cleanB64 = image.replace(/^data:image\/\w+;base64,/, '');
    const imgBuffer = Buffer.from(cleanB64, 'base64');

    // ইনফারেন্সের সুবিধার্থে ছবি সর্বোচ্চ 512px-এ প্রিপেয়ার করা
    const resizedBuffer = await sharp(imgBuffer)
      .resize(512, 512, { fit: 'inside', withoutEnlargement: true })
      .png()
      .toBuffer();

    const model = await getSegmenter();
    const dataUri = `data:image/png;base64,${resizedBuffer.toString('base64')}`;
    const output = await model(dataUri);

    const partsMap = {};

    for (const item of output) {
      const rawLabel = item.label;
      const cleanLabel = LABEL_MAP[rawLabel] || rawLabel;
      if (!cleanLabel) continue;

      const maskW = item.mask.width;
      const maskH = item.mask.height;
      const rawData = item.mask.data;

      if (!partsMap[cleanLabel]) {
        partsMap[cleanLabel] = {
          width: maskW,
          height: maskH,
          data: new Uint8Array(rawData)
        };
      } else {
        const existing = partsMap[cleanLabel].data;
        for (let i = 0; i < existing.length; i++) {
          existing[i] = Math.max(existing[i], rawData[i]);
        }
      }
    }

    let subjectData = null;
    let sW = 0, sH = 0;
    const finalParts = [];

    for (const [label, maskObj] of Object.entries(partsMap)) {
      if (label === 'Background') continue;
      if (!subjectData) {
        sW = maskObj.width;
        sH = maskObj.height;
        subjectData = new Uint8Array(maskObj.data.length);
      }
      for (let i = 0; i < maskObj.data.length; i++) {
        subjectData[i] = Math.max(subjectData[i], maskObj.data[i]);
      }
    }

    if (subjectData) {
      const subjPngBuffer = await sharp(subjectData, {
        raw: { width: sW, height: sH, channels: 1 }
      }).png().toBuffer();

      finalParts.push({
        label: 'Subject',
        mask: subjPngBuffer.toString('base64')
      });
    }

    for (const [label, maskObj] of Object.entries(partsMap)) {
      if (label === 'Background') continue;
      const pngBuffer = await sharp(maskObj.data, {
        raw: { width: maskObj.width, height: maskObj.height, channels: 1 }
      }).png().toBuffer();

      finalParts.push({
        label: label,
        mask: pngBuffer.toString('base64')
      });
    }

    if (subjectData) {
      const bgData = new Uint8Array(subjectData.length);
      for (let i = 0; i < subjectData.length; i++) {
        bgData[i] = 255 - subjectData[i];
      }
      const bgPngBuffer = await sharp(bgData, {
        raw: { width: sW, height: sH, channels: 1 }
      }).png().toBuffer();

      finalParts.push({
        label: 'Background',
        mask: bgPngBuffer.toString('base64')
      });
    }

    return res.status(200).json({
      success: true,
      parts: finalParts
    });

  } catch (err) {
    console.error('API Error:', err);
    return res.status(500).json({
      success: false,
      error: err.message || String(err)
    });
  }
}
