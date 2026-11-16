import { pipeline, env } from '@xenova/transformers';
import sharp from 'sharp';

// Vercel Lambda Cache সেটিংস
env.allowLocalModels = false;
env.useBrowserCache = false;

// গ্লোবাল ক্যাশে মডেল লোড রাখা (Cold Start কমাবে)
let segmenter = null;

async function getSegmenter() {
  if (!segmenter) {
    segmenter = await pipeline('image-segmentation', 'Xenova/segformer_b2_clothes', {
      quantized: true, // হালকা ও ফাস্ট এক্সিকিউশন
    });
  }
  return segmenter;
}

// ১৮টি লেবেলের ইউজার-ফ্রেন্ডলি ম্যাপিং
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
  // CORS হেডার (যাতে যেকোনো রিকোয়েস্ট এক্সেপ্ট করে)
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
    const { image } = req.body;
    if (!image) {
      return res.status(400).json({ success: false, error: 'image base64 is required' });
    }

    // Base64 ক্লিন ও বাফার কনভার্ট
    const cleanB64 = image.replace(/^data:image\/\w+;base64,/, '');
    const imgBuffer = Buffer.from(cleanB64, 'base64');

    // ইনপুট ছবির সাইজ নেওয়া
    const metadata = await sharp(imgBuffer).metadata();
    const origW = metadata.width || 512;
    const origH = metadata.height || 512;

    // AI ইনফারেন্স চালানো
    const model = await getSegmenter();
    // Transformers.js এর জন্য ডাটা-ইউআরআই
    const dataUri = `data:image/png;base64,${cleanB64}`;
    const output = await model(dataUri);

    // পার্টস একত্রিত করার কাঠামো
    // output একটি অ্যারে: [{ score, label, mask: RawImage }, ...]
    const partsMap = {};

    for (const item of output) {
      const rawLabel = item.label;
      const cleanLabel = LABEL_MAP[rawLabel] || rawLabel;
      if (!cleanLabel) continue;

      const maskW = item.mask.width;
      const maskH = item.mask.height;
      const rawData = item.mask.data; // Uint8ClampedArray (আলফা চ্যানেল)

      // যদি একই গ্রুপের একাধিক পার্ট থাকে (যেমন Left-arm + Right-arm -> Skin/Body), তাদের আলফা মার্চ করা
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

    // পুরো মানুষটির সাবজেক্ট মাস্ক তৈরি (সবকিছু মিলিয়ে Background বাদে)
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

    // ১. Subject যুক্ত করা
    if (subjectData) {
      const subjPngBuffer = await sharp(subjectData, {
        raw: { width: sW, height: sH, channels: 1 }
      })
      .png()
      .toBuffer();

      finalParts.push({
        label: 'Subject',
        mask: subjPngBuffer.toString('base64')
      });
    }

    // ২. প্রতিটি আলাদা পার্টস যুক্ত করা (Face, Hair, Shirt/Top, Pants/Skirt ইত্যাদি)
    for (const [label, maskObj] of Object.entries(partsMap)) {
      if (label === 'Background') continue;
      const pngBuffer = await sharp(maskObj.data, {
        raw: { width: maskObj.width, height: maskObj.height, channels: 1 }
      })
      .png()
      .toBuffer();

      finalParts.push({
        label: label,
        mask: pngBuffer.toString('base64')
      });
    }

    // ৩. Background মাস্ক (Subject এর উল্টো)
    if (subjectData) {
      const bgData = new Uint8Array(subjectData.length);
      for (let i = 0; i < subjectData.length; i++) {
        bgData[i] = 255 - subjectData[i];
      }
      const bgPngBuffer = await sharp(bgData, {
        raw: { width: sW, height: sH, channels: 1 }
      })
      .png()
      .toBuffer();

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
    console.error('Segmentation error:', err);
    return res.status(500).json({
      success: false,
      error: err.message || 'Failed to segment image'
    });
  }
}
