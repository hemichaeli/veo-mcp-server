import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { SSEServerTransport } from "@modelcontextprotocol/sdk/server/sse.js";
import express from "express";
import { z } from "zod";

// --- Configuration ---

const GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta";
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || "";
const PORT = parseInt(process.env.PORT || "3000");

// Official Veo 3.1 model family tiers
const MODELS = [
  { id: "veo-3.1-generate-preview", name: "Veo 3.1 Pro", description: "Premium quality. 4K/1080p/720p, native audio, reference images, first+last frame, video extension. Best for cinematic and enterprise work." },
  { id: "veo-3.1-fast-generate-preview", name: "Veo 3.1 Fast", description: "Balanced speed-to-cost. 1080p/720p, native audio. Good for rapid iteration, A/B testing, social media." },
  { id: "veo-3.1-lite-generate-preview", name: "Veo 3.1 Lite", description: "Most cost-effective (<50% of Fast price). 1080p/720p. No 4K, no extension. High-volume apps." },
  { id: "veo-3-generate-preview", name: "Veo 3", description: "Previous gen. Native audio, 1080p/720p." },
  { id: "veo-2-generate-001", name: "Veo 2", description: "Stable. No native audio. 720p/1080p. Supports enhancePrompt." },
] as const;

// --- API Helper ---

async function geminiRequest(
  path: string,
  method: "GET" | "POST" = "GET",
  body?: Record<string, unknown>
): Promise<unknown> {
  const url = `${GEMINI_API_BASE}/${path}`;
  const headers: Record<string, string> = {
    "x-goog-api-key": GEMINI_API_KEY,
    "Content-Type": "application/json",
  };

  const res = await fetch(url, {
    method,
    headers,
    ...(body ? { body: JSON.stringify(body) } : {}),
  });

  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(`Gemini API error ${res.status}: ${errorText}`);
  }

  return res.json();
}

async function fetchImageAsBase64(url: string): Promise<{ base64: string; mimeType: string }> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Could not fetch image from URL: ${res.status} ${res.statusText}`);
  }
  const buf = Buffer.from(await res.arrayBuffer());
  const mimeType = res.headers.get("content-type") || "image/png";
  return { base64: buf.toString("base64"), mimeType };
}

// --- Server Setup ---

const server = new McpServer({
  name: "veo-mcp-server",
  version: "2.1.0",
});

// --- Shared helpers ---

function buildParameters(params: {
  aspect_ratio?: string;
  resolution?: string;
  duration_seconds?: number;
  sample_count?: number;
  person_generation?: string;
  generate_audio?: boolean;
  negative_prompt?: string;
  enhance_prompt?: boolean;
  compression_quality?: string;
  seed?: number;
}): Record<string, unknown> {
  const p: Record<string, unknown> = {};
  if (params.aspect_ratio) p.aspectRatio = params.aspect_ratio;
  if (params.resolution) p.resolution = params.resolution;
  if (params.duration_seconds) p.durationSeconds = params.duration_seconds;
  if (params.sample_count) p.sampleCount = params.sample_count;
  if (params.person_generation) p.personGeneration = params.person_generation;
  if (params.generate_audio !== undefined) p.generateAudio = params.generate_audio;
  if (params.negative_prompt) p.negativePrompt = params.negative_prompt;
  if (params.enhance_prompt !== undefined) p.enhancePrompt = params.enhance_prompt;
  if (params.compression_quality) p.compressionQuality = params.compression_quality;
  if (params.seed !== undefined) p.seed = params.seed;
  return p;
}

function checkApiKey() {
  if (!GEMINI_API_KEY) {
    return {
      content: [{ type: "text" as const, text: "Error: GEMINI_API_KEY environment variable is not set." }],
      isError: true as const,
    };
  }
  return null;
}

// --- Tool: List Models ---

server.registerTool(
  "veo_list_models",
  {
    title: "List Veo Models",
    description: "List all available Google Veo video generation models with their tiers, capabilities, and model IDs.",
    inputSchema: {},
    annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
  },
  async () => {
    const modelList = MODELS.map(
      (m) => `- **${m.name}** (\`${m.id}\`): ${m.description}`
    ).join("\n");
    return { content: [{ type: "text", text: `Available Veo Models:\n\n${modelList}` }] };
  }
);

// --- Tool: Generate Video (Text-to-Video) ---

const GenerateVideoSchema = z.object({
  prompt: z.string().min(1).max(5000).describe("Text prompt. Be specific about camera angles, lighting, mood, audio, action, dialogue."),
  model: z.string().default("veo-3.1-generate-preview").describe("Model ID. Default: veo-3.1-generate-preview (Veo 3.1 Pro). Options: veo-3.1-generate-preview, veo-3.1-fast-generate-preview, veo-3.1-lite-generate-preview, veo-3-generate-preview, veo-2-generate-001"),
  aspect_ratio: z.enum(["16:9", "9:16"]).default("16:9").describe("16:9 (landscape) or 9:16 (portrait)"),
  resolution: z.enum(["720p", "1080p", "4k"]).default("1080p").describe("720p, 1080p, or 4k (4k: Veo 3.1 Pro only)"),
  duration_seconds: z.number().int().min(4).max(8).default(8).optional().describe("Duration: 4, 5, 6, or 8 seconds"),
  sample_count: z.number().int().min(1).max(4).default(1).optional().describe("Number of video variants (1-4)"),
  person_generation: z.enum(["dont_allow", "allow_adult"]).default("allow_adult").optional().describe("People generation control"),
  generate_audio: z.boolean().default(true).optional().describe("Native synchronized audio (Veo 3+ only)"),
  negative_prompt: z.string().max(2000).optional().describe("Content to prevent from appearing"),
  enhance_prompt: z.boolean().optional().describe("Auto-enhance prompt (Veo 2 only)"),
  compression_quality: z.string().optional().describe("Video compression quality"),
  seed: z.number().int().optional().describe("Seed for determinism (Veo 3+)"),
  wait: z.boolean().default(false).optional().describe("Poll until ready (up to 6 min). Default: return operation ID immediately."),
}).strict();

server.registerTool(
  "veo_generate_video",
  {
    title: "Generate Video from Text",
    description: `Generate video from text prompt using Google Veo. Default model is Veo 3.1 Pro (best quality, 4K, native audio). Supports negative prompts, audio control, seed, multiple variants. Set wait=true to block until video is ready (11s-6min), or false (default) to get operation ID for async polling.`,
    inputSchema: GenerateVideoSchema,
    annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: false, openWorldHint: true },
  },
  async (params) => {
    const err = checkApiKey();
    if (err) return err;

    const parameters = buildParameters(params);
    const body = { instances: [{ prompt: params.prompt }], parameters };

    const result = await geminiRequest(
      `models/${params.model}:predictLongRunning`, "POST", body
    ) as { name: string };

    if (!params.wait) {
      return {
        content: [{
          type: "text",
          text: `Video generation submitted.\n\nOperation: \`${result.name}\`\nModel: ${params.model}\nResolution: ${params.resolution}\nAspect: ${params.aspect_ratio}\nAudio: ${params.generate_audio !== false ? "Yes" : "No"}\n${params.negative_prompt ? `Negative: ${params.negative_prompt}\n` : ""}\nUse \`veo_check_operation\` to check status (11s-6min).`,
        }],
      };
    }
    return await pollOperation(result.name);
  }
);

// --- Tool: Image to Video (First Frame + Optional Last Frame) ---

const ImageToVideoSchema = z.object({
  prompt: z.string().min(1).max(5000).describe("Text prompt describing desired motion/action"),
  image_url: z.string().url().describe("URL of starting (first) frame image"),
  last_frame_url: z.string().url().optional().describe("Optional: URL of ending (last) frame image. Veo interpolates motion between first and last frame."),
  model: z.string().default("veo-3.1-generate-preview").describe("Model ID (default: Veo 3.1 Pro)"),
  aspect_ratio: z.enum(["16:9", "9:16"]).default("16:9").describe("Aspect ratio"),
  resolution: z.enum(["720p", "1080p", "4k"]).default("1080p").describe("Resolution"),
  generate_audio: z.boolean().default(true).optional().describe("Native audio"),
  negative_prompt: z.string().max(2000).optional().describe("Content to prevent"),
  sample_count: z.number().int().min(1).max(4).default(1).optional().describe("Variants (1-4)"),
  seed: z.number().int().optional().describe("Seed"),
  wait: z.boolean().default(false).optional().describe("Poll until done"),
}).strict();

server.registerTool(
  "veo_image_to_video",
  {
    title: "Generate Video from Image (First + Last Frame)",
    description: `Animate an image into video, or interpolate between two images.

- First frame only: animates from the starting image with described motion
- First + Last frame: smooth cinematic transition between two images

Supports all Veo models. Combine with text prompt for motion direction.`,
    inputSchema: ImageToVideoSchema,
    annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: false, openWorldHint: true },
  },
  async (params) => {
    const err = checkApiKey();
    if (err) return err;

    const firstImage = await fetchImageAsBase64(params.image_url);
    const instance: Record<string, unknown> = {
      prompt: params.prompt,
      image: { bytesBase64Encoded: firstImage.base64, mimeType: firstImage.mimeType },
    };

    if (params.last_frame_url) {
      const lastImage = await fetchImageAsBase64(params.last_frame_url);
      instance.lastFrame = { inlineData: { mimeType: lastImage.mimeType, data: lastImage.base64 } };
    }

    const parameters = buildParameters({
      aspect_ratio: params.aspect_ratio, resolution: params.resolution,
      generate_audio: params.generate_audio, negative_prompt: params.negative_prompt,
      sample_count: params.sample_count, seed: params.seed,
    });

    const body = { instances: [instance], parameters };
    const result = await geminiRequest(`models/${params.model}:predictLongRunning`, "POST", body) as { name: string };
    const mode = params.last_frame_url ? "First+Last frame interpolation" : "First frame animation";

    if (!params.wait) {
      return { content: [{ type: "text", text: `${mode} submitted.\n\nOperation: \`${result.name}\`\n\nUse \`veo_check_operation\` to check status.` }] };
    }
    return await pollOperation(result.name);
  }
);

// --- Tool: Extend Video ---

const ExtendVideoSchema = z.object({
  prompt: z.string().min(1).max(5000).describe("What should happen in the extension"),
  video_uri: z.string().describe("URI of previously generated Veo video to extend"),
  model: z.string().default("veo-3.1-generate-preview").describe("Model ID (default: Veo 3.1 Pro). Note: Lite does not support extension."),
  generate_audio: z.boolean().default(true).optional().describe("Audio for extension"),
  negative_prompt: z.string().max(2000).optional().describe("Content to prevent"),
  seed: z.number().int().optional().describe("Seed"),
  wait: z.boolean().default(false).optional().describe("Poll until done"),
}).strict();

server.registerTool(
  "veo_extend_video",
  {
    title: "Extend a Generated Video",
    description: `Extend a Veo-generated video by ~7 seconds, up to 20 times (max ~148 seconds total). Continues from the last second of the previous clip for visual continuity. Extension is 720p only. Not supported by Veo 3.1 Lite.`,
    inputSchema: ExtendVideoSchema,
    annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: false, openWorldHint: true },
  },
  async (params) => {
    const err = checkApiKey();
    if (err) return err;

    const parameters = buildParameters({
      resolution: "720p", generate_audio: params.generate_audio,
      negative_prompt: params.negative_prompt, seed: params.seed,
    });

    const body = {
      instances: [{ prompt: params.prompt, video: { uri: params.video_uri, mimeType: "video/mp4" } }],
      parameters,
    };

    const result = await geminiRequest(`models/${params.model}:predictLongRunning`, "POST", body) as { name: string };

    if (!params.wait) {
      return { content: [{ type: "text", text: `Video extension submitted.\n\nOperation: \`${result.name}\`\n\nUse \`veo_check_operation\` to check status.` }] };
    }
    return await pollOperation(result.name);
  }
);

// --- Tool: Generate with Reference Images ---

const ReferenceImageSchema = z.object({
  prompt: z.string().min(1).max(5000).describe("Text prompt describing the video"),
  reference_image_urls: z.array(z.string().url()).min(1).max(3).describe("1-3 image URLs for character/style/scene guidance"),
  reference_type: z.enum(["asset", "style"]).default("asset").optional().describe("'asset': subject/character consistency (up to 3 images). 'style': artistic style transfer (1 image, Veo 2 exp)."),
  model: z.string().default("veo-3.1-generate-preview").describe("Model ID (default: Veo 3.1 Pro)"),
  aspect_ratio: z.enum(["16:9", "9:16"]).default("16:9").describe("Aspect ratio"),
  resolution: z.enum(["720p", "1080p", "4k"]).default("1080p").describe("Resolution"),
  generate_audio: z.boolean().default(true).optional().describe("Native audio"),
  negative_prompt: z.string().max(2000).optional().describe("Content to prevent"),
  sample_count: z.number().int().min(1).max(4).default(1).optional().describe("Variants (1-4)"),
  seed: z.number().int().optional().describe("Seed"),
  wait: z.boolean().default(false).optional().describe("Poll until done"),
}).strict();

server.registerTool(
  "veo_generate_with_references",
  {
    title: "Generate Video with Reference Images",
    description: `Guide video generation with up to 3 reference images.

Types:
- 'asset' (default): Subject images for character/product consistency across shots. Up to 3 images. Veo 3.1 Pro/Fast.
- 'style': Apply artistic style from 1 image. Veo 2 experimental.`,
    inputSchema: ReferenceImageSchema,
    annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: false, openWorldHint: true },
  },
  async (params) => {
    const err = checkApiKey();
    if (err) return err;

    const refType = params.reference_type || "asset";
    const referenceImages: Array<{ image: { bytesBase64Encoded: string; mimeType: string }; referenceType: string }> = [];

    for (const url of params.reference_image_urls) {
      const img = await fetchImageAsBase64(url);
      referenceImages.push({
        image: { bytesBase64Encoded: img.base64, mimeType: img.mimeType },
        referenceType: refType,
      });
    }

    const parameters = buildParameters({
      aspect_ratio: params.aspect_ratio, resolution: params.resolution,
      generate_audio: params.generate_audio, negative_prompt: params.negative_prompt,
      sample_count: params.sample_count, seed: params.seed,
    });
    parameters.referenceImages = referenceImages;

    const body = { instances: [{ prompt: params.prompt }], parameters };
    const result = await geminiRequest(`models/${params.model}:predictLongRunning`, "POST", body) as { name: string };

    if (!params.wait) {
      return { content: [{ type: "text", text: `Video with ${params.reference_image_urls.length} ${refType} reference(s) submitted.\n\nOperation: \`${result.name}\`\n\nUse \`veo_check_operation\` to check status.` }] };
    }
    return await pollOperation(result.name);
  }
);

// --- Tool: Check Operation ---

const CheckOperationSchema = z.object({
  operation_name: z.string().min(1).describe("Operation name from generate/extend call (e.g. 'operations/...')"),
}).strict();

server.registerTool(
  "veo_check_operation",
  {
    title: "Check Video Generation Status",
    description: `Check status of a Veo operation. Returns processing/completed/error with video download URIs. Videos stored 2 days. All videos have SynthID watermarks.`,
    inputSchema: CheckOperationSchema,
    annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: true },
  },
  async (params) => {
    const err = checkApiKey();
    if (err) return err;

    const status = await geminiRequest(params.operation_name) as {
      done?: boolean;
      response?: { generateVideoResponse?: { generatedSamples?: Array<{ video?: { uri?: string } }> } };
      error?: { message?: string };
    };

    if (status.error) {
      return { content: [{ type: "text", text: `Error: ${status.error.message || "Unknown error"}` }], isError: true };
    }

    if (!status.done) {
      return { content: [{ type: "text", text: `Status: **Processing**\n\nStill generating. Check again in 10-30 seconds.\n\nOperation: \`${params.operation_name}\`` }] };
    }

    const samples = status.response?.generateVideoResponse?.generatedSamples || [];
    if (samples.length === 0) {
      return { content: [{ type: "text", text: "Status: **Completed** but no videos returned. Content may have been filtered by safety checks." }] };
    }

    const videoLines = samples.map((s, i) => `Video ${i + 1}: ${s.video?.uri || "No URI"}`);
    return { content: [{ type: "text", text: `Status: **Completed**\n\n${videoLines.join("\n")}\n\nVideos expire after 2 days. All videos include SynthID watermarks.` }] };
  }
);

// --- Poll helper ---

async function pollOperation(operationName: string) {
  const maxWaitMs = 7 * 60 * 1000;
  const pollIntervalMs = 10_000;
  const startTime = Date.now();

  while (Date.now() - startTime < maxWaitMs) {
    await new Promise((r) => setTimeout(r, pollIntervalMs));
    const status = await geminiRequest(operationName) as {
      done?: boolean;
      response?: { generateVideoResponse?: { generatedSamples?: Array<{ video?: { uri?: string } }> } };
      error?: { message?: string };
    };

    if (status.error) {
      return { content: [{ type: "text" as const, text: `Error: ${status.error.message || "Unknown error"}` }], isError: true as const };
    }

    if (status.done) {
      const samples = status.response?.generateVideoResponse?.generatedSamples || [];
      if (samples.length === 0) {
        return { content: [{ type: "text" as const, text: "Completed but no videos returned. Content may have been filtered." }] };
      }
      const videoLines = samples.map((s, i) => `Video ${i + 1}: ${s.video?.uri || "No URI"}`);
      return { content: [{ type: "text" as const, text: `Video generation complete!\n\n${videoLines.join("\n")}\n\nVideos expire after 2 days. Download promptly.` }] };
    }
  }

  return { content: [{ type: "text" as const, text: `Still processing after 7 min. Operation: \`${operationName}\`\nCheck again with veo_check_operation.` }] };
}

// --- SSE Transport ---

const app = express();
const transports = new Map<string, SSEServerTransport>();

app.get("/sse", async (_req, res) => {
  const transport = new SSEServerTransport("/messages", res);
  transports.set(transport.sessionId, transport);
  res.on("close", () => { transports.delete(transport.sessionId); });
  await server.connect(transport);
});

app.post("/messages", async (req, res) => {
  const sessionId = req.query.sessionId as string;
  const transport = transports.get(sessionId);
  if (!transport) { res.status(404).json({ error: "Session not found or expired" }); return; }
  await transport.handlePostMessage(req, res);
});

app.get("/health", (_req, res) => {
  res.json({ status: "ok", service: "veo-mcp-server", version: "2.1.0" });
});

app.listen(PORT, () => {
  console.log(`Veo MCP Server v2.1.0 running on port ${PORT}`);
  console.log(`SSE endpoint: http://localhost:${PORT}/sse`);
  console.log(`API Key configured: ${GEMINI_API_KEY ? "Yes" : "No"}`);
});
