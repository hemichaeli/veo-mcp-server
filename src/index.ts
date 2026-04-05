import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { SSEServerTransport } from "@modelcontextprotocol/sdk/server/sse.js";
import express from "express";
import { z } from "zod";

// --- Configuration ---

const GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta";
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || "";
const PORT = parseInt(process.env.PORT || "3000");

const MODELS = [
  { id: "veo-3.1-generate-preview", name: "Veo 3.1", description: "Best quality, 720p/1080p/4k, native audio, reference images, first+last frame" },
  { id: "veo-3.1-fast-generate-preview", name: "Veo 3.1 Fast", description: "Faster generation, lower cost, good quality, native audio" },
  { id: "veo-3.1-lite-generate-preview", name: "Veo 3.1 Lite", description: "Lowest cost Veo 3.1 variant, released March 2026" },
  { id: "veo-3-generate-preview", name: "Veo 3", description: "Previous gen, native audio, 720p/1080p" },
  { id: "veo-2-generate-001", name: "Veo 2", description: "Stable, no native audio, 720p/1080p, supports enhancePrompt" },
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
  version: "2.0.0",
});

// --- Tool: List Models ---

server.registerTool(
  "veo_list_models",
  {
    title: "List Veo Models",
    description: "List all available Google Veo video generation models with their capabilities, IDs, and descriptions.",
    inputSchema: {},
    annotations: {
      readOnlyHint: true,
      destructiveHint: false,
      idempotentHint: true,
      openWorldHint: false,
    },
  },
  async () => {
    const modelList = MODELS.map(
      (m) => `- **${m.name}** (\`${m.id}\`): ${m.description}`
    ).join("\n");

    return {
      content: [{ type: "text", text: `Available Veo Models:\n\n${modelList}` }],
    };
  }
);

// --- Shared parameter helpers ---

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

// --- Tool: Generate Video (Text-to-Video) ---

const GenerateVideoSchema = z.object({
  prompt: z.string().min(1).max(5000).describe("Text prompt describing the video. Be specific about camera angles, lighting, mood, audio, action."),
  model: z.string().default("veo-3.1-generate-preview").describe("Model ID. Options: veo-3.1-generate-preview, veo-3.1-fast-generate-preview, veo-3.1-lite-generate-preview, veo-3-generate-preview, veo-2-generate-001"),
  aspect_ratio: z.enum(["16:9", "9:16"]).default("16:9").describe("Aspect ratio: 16:9 (landscape) or 9:16 (portrait)"),
  resolution: z.enum(["720p", "1080p", "4k"]).default("1080p").describe("Resolution: 720p, 1080p, or 4k (4k only for Veo 3.1)"),
  duration_seconds: z.number().int().min(4).max(8).default(8).optional().describe("Video duration in seconds (4, 5, 6, or 8)"),
  sample_count: z.number().int().min(1).max(4).default(1).optional().describe("Number of video variants to generate (1-4)"),
  person_generation: z.enum(["dont_allow", "allow_adult"]).default("allow_adult").optional().describe("Whether to allow generating people"),
  generate_audio: z.boolean().default(true).optional().describe("Generate native synchronized audio (Veo 3+ only)"),
  negative_prompt: z.string().max(2000).optional().describe("Describe content to prevent from appearing in the video"),
  enhance_prompt: z.boolean().optional().describe("Auto-enhance the prompt (Veo 2 models only)"),
  compression_quality: z.string().optional().describe("Video compression quality setting"),
  seed: z.number().int().optional().describe("Seed for slightly improved determinism (Veo 3+ only)"),
  wait: z.boolean().default(false).optional().describe("If true, poll until video is ready (may take up to 6 min). If false, return operation ID immediately."),
}).strict();

server.registerTool(
  "veo_generate_video",
  {
    title: "Generate Video from Text",
    description: `Generate a video from a text prompt using Google Veo. Submits a video generation request to the Gemini API. By default returns the operation name immediately (async). Set wait=true to poll until the video is ready (11s to 6 min). Supports negative prompts, audio control, seed, multiple variants, and all Veo models.`,
    inputSchema: GenerateVideoSchema,
    annotations: {
      readOnlyHint: false,
      destructiveHint: false,
      idempotentHint: false,
      openWorldHint: true,
    },
  },
  async (params) => {
    const err = checkApiKey();
    if (err) return err;

    const parameters = buildParameters(params);

    const body = {
      instances: [{ prompt: params.prompt }],
      parameters,
    };

    const result = await geminiRequest(
      `models/${params.model}:predictLongRunning`,
      "POST",
      body
    ) as { name: string };

    if (!params.wait) {
      return {
        content: [{
          type: "text",
          text: `Video generation submitted.\n\nOperation: \`${result.name}\`\nModel: ${params.model}\nResolution: ${params.resolution}\nAspect Ratio: ${params.aspect_ratio}\nAudio: ${params.generate_audio !== false ? "Yes" : "No"}\n${params.negative_prompt ? `Negative Prompt: ${params.negative_prompt}\n` : ""}\nUse \`veo_check_operation\` to check status. Generation typically takes 11 seconds to 6 minutes.`,
        }],
      };
    }

    return await pollOperation(result.name);
  }
);

// --- Tool: Image to Video (First Frame + Optional Last Frame) ---

const ImageToVideoSchema = z.object({
  prompt: z.string().min(1).max(5000).describe("Text prompt describing the desired video motion/action"),
  image_url: z.string().url().describe("URL of the image to use as the starting (first) frame"),
  last_frame_url: z.string().url().optional().describe("Optional URL of the image to use as the ending (last) frame. Veo will interpolate motion between first and last frame."),
  model: z.string().default("veo-3.1-generate-preview").describe("Model ID"),
  aspect_ratio: z.enum(["16:9", "9:16"]).default("16:9").describe("Aspect ratio"),
  resolution: z.enum(["720p", "1080p", "4k"]).default("1080p").describe("Resolution"),
  generate_audio: z.boolean().default(true).optional().describe("Generate native audio"),
  negative_prompt: z.string().max(2000).optional().describe("Content to prevent from appearing"),
  sample_count: z.number().int().min(1).max(4).default(1).optional().describe("Number of variants (1-4)"),
  seed: z.number().int().optional().describe("Seed for determinism"),
  wait: z.boolean().default(false).optional().describe("Poll until done"),
}).strict();

server.registerTool(
  "veo_image_to_video",
  {
    title: "Generate Video from Image (First + Last Frame)",
    description: `Generate a video using an image as the starting frame, with optional last frame for interpolation.

Fetches image(s) from URL(s) and uses them to control video generation:
- First frame only: animates from the starting image
- First + Last frame: interpolates smooth motion between two images (great for transitions)

Combine with a text prompt to describe the desired motion.`,
    inputSchema: ImageToVideoSchema,
    annotations: {
      readOnlyHint: false,
      destructiveHint: false,
      idempotentHint: false,
      openWorldHint: true,
    },
  },
  async (params) => {
    const err = checkApiKey();
    if (err) return err;

    const firstImage = await fetchImageAsBase64(params.image_url);

    const instance: Record<string, unknown> = {
      prompt: params.prompt,
      image: {
        bytesBase64Encoded: firstImage.base64,
        mimeType: firstImage.mimeType,
      },
    };

    // Add last frame if provided
    if (params.last_frame_url) {
      const lastImage = await fetchImageAsBase64(params.last_frame_url);
      instance.lastFrame = {
        inlineData: {
          mimeType: lastImage.mimeType,
          data: lastImage.base64,
        },
      };
    }

    const parameters = buildParameters({
      aspect_ratio: params.aspect_ratio,
      resolution: params.resolution,
      generate_audio: params.generate_audio,
      negative_prompt: params.negative_prompt,
      sample_count: params.sample_count,
      seed: params.seed,
    });

    const body = {
      instances: [instance],
      parameters,
    };

    const result = await geminiRequest(
      `models/${params.model}:predictLongRunning`,
      "POST",
      body
    ) as { name: string };

    const mode = params.last_frame_url ? "First + Last frame interpolation" : "First frame animation";

    if (!params.wait) {
      return {
        content: [{
          type: "text",
          text: `${mode} submitted.\n\nOperation: \`${result.name}\`\n\nUse \`veo_check_operation\` to check status.`,
        }],
      };
    }

    return await pollOperation(result.name);
  }
);

// --- Tool: Extend Video ---

const ExtendVideoSchema = z.object({
  prompt: z.string().min(1).max(5000).describe("Prompt describing what should happen in the extension"),
  video_uri: z.string().describe("URI of a previously generated Veo video to extend"),
  model: z.string().default("veo-3.1-generate-preview").describe("Model ID"),
  generate_audio: z.boolean().default(true).optional().describe("Generate audio for extension"),
  negative_prompt: z.string().max(2000).optional().describe("Content to prevent"),
  seed: z.number().int().optional().describe("Seed"),
  wait: z.boolean().default(false).optional().describe("Poll until done"),
}).strict();

server.registerTool(
  "veo_extend_video",
  {
    title: "Extend a Generated Video",
    description: `Extend a previously generated Veo video with new content. Takes the URI of a Veo-generated video and creates a new clip continuing from the last second. Chain multiple extensions for longer narratives (60+ seconds). Extension is limited to 720p.`,
    inputSchema: ExtendVideoSchema,
    annotations: {
      readOnlyHint: false,
      destructiveHint: false,
      idempotentHint: false,
      openWorldHint: true,
    },
  },
  async (params) => {
    const err = checkApiKey();
    if (err) return err;

    const parameters = buildParameters({
      resolution: "720p",
      generate_audio: params.generate_audio,
      negative_prompt: params.negative_prompt,
      seed: params.seed,
    });

    const body = {
      instances: [{
        prompt: params.prompt,
        video: {
          uri: params.video_uri,
          mimeType: "video/mp4",
        },
      }],
      parameters,
    };

    const result = await geminiRequest(
      `models/${params.model}:predictLongRunning`,
      "POST",
      body
    ) as { name: string };

    if (!params.wait) {
      return {
        content: [{
          type: "text",
          text: `Video extension submitted.\n\nOperation: \`${result.name}\`\n\nUse \`veo_check_operation\` to check status.`,
        }],
      };
    }

    return await pollOperation(result.name);
  }
);

// --- Tool: Generate with Reference Images ---

const ReferenceImageSchema = z.object({
  prompt: z.string().min(1).max(5000).describe("Text prompt describing the video"),
  reference_image_urls: z.array(z.string().url()).min(1).max(3).describe("1-3 URLs of reference images for character/style/scene consistency"),
  reference_type: z.enum(["asset", "style"]).default("asset").optional().describe("Reference type: 'asset' for subject/character consistency, 'style' for artistic style transfer"),
  model: z.string().default("veo-3.1-generate-preview").describe("Model ID"),
  aspect_ratio: z.enum(["16:9", "9:16"]).default("16:9").describe("Aspect ratio"),
  resolution: z.enum(["720p", "1080p", "4k"]).default("1080p").describe("Resolution"),
  generate_audio: z.boolean().default(true).optional().describe("Generate native audio"),
  negative_prompt: z.string().max(2000).optional().describe("Content to prevent"),
  sample_count: z.number().int().min(1).max(4).default(1).optional().describe("Number of variants"),
  seed: z.number().int().optional().describe("Seed"),
  wait: z.boolean().default(false).optional().describe("Poll until done"),
}).strict();

server.registerTool(
  "veo_generate_with_references",
  {
    title: "Generate Video with Reference Images",
    description: `Generate a video guided by up to 3 reference images.

Two reference types:
- 'asset' (default): Subject images for character/product consistency across shots
- 'style': Style image to apply artistic direction (1 image, Veo 2 experimental)

Supports Veo 3.1 and Veo 2 (experimental).`,
    inputSchema: ReferenceImageSchema,
    annotations: {
      readOnlyHint: false,
      destructiveHint: false,
      idempotentHint: false,
      openWorldHint: true,
    },
  },
  async (params) => {
    const err = checkApiKey();
    if (err) return err;

    const referenceImages: Array<{ image: { bytesBase64Encoded: string; mimeType: string }; referenceType: string }> = [];
    const refType = params.reference_type || "asset";

    for (const url of params.reference_image_urls) {
      const img = await fetchImageAsBase64(url);
      referenceImages.push({
        image: { bytesBase64Encoded: img.base64, mimeType: img.mimeType },
        referenceType: refType,
      });
    }

    const parameters = buildParameters({
      aspect_ratio: params.aspect_ratio,
      resolution: params.resolution,
      generate_audio: params.generate_audio,
      negative_prompt: params.negative_prompt,
      sample_count: params.sample_count,
      seed: params.seed,
    });
    parameters.referenceImages = referenceImages;

    const body = {
      instances: [{ prompt: params.prompt }],
      parameters,
    };

    const result = await geminiRequest(
      `models/${params.model}:predictLongRunning`,
      "POST",
      body
    ) as { name: string };

    if (!params.wait) {
      return {
        content: [{
          type: "text",
          text: `Video with ${params.reference_image_urls.length} ${refType} reference(s) submitted.\n\nOperation: \`${result.name}\`\n\nUse \`veo_check_operation\` to check status.`,
        }],
      };
    }

    return await pollOperation(result.name);
  }
);

// --- Tool: Check Operation Status ---

const CheckOperationSchema = z.object({
  operation_name: z.string().min(1).describe("The operation name returned from a generate/extend call (e.g., 'operations/...')"),
}).strict();

server.registerTool(
  "veo_check_operation",
  {
    title: "Check Video Generation Status",
    description: `Check the status of a Veo video generation operation. Returns processing/completed/error status with video download URIs when ready. Videos are stored for 2 days on Google servers. All videos include SynthID watermarks.`,
    inputSchema: CheckOperationSchema,
    annotations: {
      readOnlyHint: true,
      destructiveHint: false,
      idempotentHint: true,
      openWorldHint: true,
    },
  },
  async (params) => {
    const err = checkApiKey();
    if (err) return err;

    const status = await geminiRequest(params.operation_name) as {
      done?: boolean;
      response?: {
        generateVideoResponse?: {
          generatedSamples?: Array<{ video?: { uri?: string } }>;
        };
      };
      error?: { message?: string };
    };

    if (status.error) {
      return {
        content: [{ type: "text", text: `Error: ${status.error.message || "Unknown error"}` }],
        isError: true,
      };
    }

    if (!status.done) {
      return {
        content: [{
          type: "text",
          text: `Status: **Processing**\n\nStill generating. Check again in 10-30 seconds.\n\nOperation: \`${params.operation_name}\``,
        }],
      };
    }

    const samples = status.response?.generateVideoResponse?.generatedSamples || [];
    if (samples.length === 0) {
      return {
        content: [{
          type: "text",
          text: "Status: **Completed** but no videos returned. Content may have been filtered by safety checks.",
        }],
      };
    }

    const videoLines = samples.map((s, i) => `Video ${i + 1}: ${s.video?.uri || "No URI"}`);

    return {
      content: [{
        type: "text",
        text: `Status: **Completed**\n\n${videoLines.join("\n")}\n\nVideos expire after 2 days. Download before removal. All videos include SynthID watermarks.`,
      }],
    };
  }
);

// --- Shared Poll Helper ---

async function pollOperation(operationName: string) {
  const maxWaitMs = 7 * 60 * 1000;
  const pollIntervalMs = 10_000;
  const startTime = Date.now();

  while (Date.now() - startTime < maxWaitMs) {
    await new Promise((r) => setTimeout(r, pollIntervalMs));
    const status = await geminiRequest(operationName) as {
      done?: boolean;
      response?: {
        generateVideoResponse?: {
          generatedSamples?: Array<{ video?: { uri?: string } }>;
        };
      };
      error?: { message?: string };
    };

    if (status.error) {
      return {
        content: [{ type: "text" as const, text: `Error: ${status.error.message || "Unknown error"}` }],
        isError: true as const,
      };
    }

    if (status.done) {
      const samples = status.response?.generateVideoResponse?.generatedSamples || [];
      if (samples.length === 0) {
        return {
          content: [{ type: "text" as const, text: "Completed but no videos returned. Content may have been filtered." }],
        };
      }

      const videoLines = samples.map((s, i) => `Video ${i + 1}: ${s.video?.uri || "No URI"}`);
      return {
        content: [{
          type: "text" as const,
          text: `Video generation complete!\n\n${videoLines.join("\n")}\n\nVideos expire after 2 days. Download promptly.`,
        }],
      };
    }
  }

  return {
    content: [{
      type: "text" as const,
      text: `Still processing after 7 minutes. Operation: \`${operationName}\`\nCheck again with veo_check_operation.`,
    }],
  };
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
  if (!transport) {
    res.status(404).json({ error: "Session not found or expired" });
    return;
  }
  await transport.handlePostMessage(req, res);
});

app.get("/health", (_req, res) => {
  res.json({ status: "ok", service: "veo-mcp-server", version: "2.0.0" });
});

app.listen(PORT, () => {
  console.log(`Veo MCP Server v2.0.0 running on port ${PORT}`);
  console.log(`SSE endpoint: http://localhost:${PORT}/sse`);
  console.log(`API Key configured: ${GEMINI_API_KEY ? "Yes" : "No"}`);
});
