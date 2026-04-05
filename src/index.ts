import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { SSEServerTransport } from "@modelcontextprotocol/sdk/server/sse.js";
import express from "express";
import { z } from "zod";

// --- Configuration ---

const GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta";
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || "";
const PORT = parseInt(process.env.PORT || "3000");

const MODELS = [
  { id: "veo-3.1-generate-preview", name: "Veo 3.1", description: "Best quality, 720p/1080p/4k, native audio, reference images" },
  { id: "veo-3.1-fast-generate-preview", name: "Veo 3.1 Fast", description: "Faster generation, lower cost, good quality" },
  { id: "veo-3-generate-preview", name: "Veo 3", description: "Previous gen, native audio, 720p/1080p" },
  { id: "veo-2-generate-001", name: "Veo 2", description: "Stable, no native audio, 720p/1080p" },
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

// --- Server Setup ---

const server = new McpServer({
  name: "veo-mcp-server",
  version: "1.0.0",
});

// --- Tool: List Models ---

server.registerTool(
  "veo_list_models",
  {
    title: "List Veo Models",
    description: "List all available Google Veo video generation models with their capabilities. Returns model IDs, names, and descriptions for all supported Veo variants.",
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
      content: [
        {
          type: "text",
          text: `Available Veo Models:\n\n${modelList}`,
        },
      ],
    };
  }
);

// --- Tool: Generate Video (Text-to-Video) ---

const GenerateVideoSchema = z.object({
  prompt: z.string().min(1).max(5000).describe("Text prompt describing the video to generate. Be specific about camera angles, lighting, mood, audio elements."),
  model: z.string().default("veo-3.1-generate-preview").describe("Model ID. Default: veo-3.1-generate-preview"),
  aspect_ratio: z.enum(["16:9", "9:16"]).default("16:9").describe("Aspect ratio: 16:9 (landscape) or 9:16 (portrait)"),
  resolution: z.enum(["720p", "1080p", "4k"]).default("1080p").describe("Resolution: 720p, 1080p, or 4k (4k only for Veo 3.1)"),
  duration_seconds: z.number().int().min(5).max(8).default(8).optional().describe("Video duration in seconds (5-8)"),
  number_of_videos: z.number().int().min(1).max(4).default(1).optional().describe("Number of videos to generate (1-4)"),
  person_generation: z.enum(["dont_allow", "allow_adult"]).default("allow_adult").optional().describe("Whether to allow generating people"),
  include_audio: z.boolean().default(true).optional().describe("Include native audio generation (Veo 3+ only)"),
  seed: z.number().int().optional().describe("Seed for slightly improved determinism (Veo 3+ only)"),
  wait: z.boolean().default(false).optional().describe("If true, poll until video is ready (may take up to 6 minutes). If false, return operation ID immediately."),
}).strict();

server.registerTool(
  "veo_generate_video",
  {
    title: "Generate Video from Text",
    description: `Generate a video from a text prompt using Google Veo. Submits a video generation request to the Gemini API. By default returns the operation name immediately (async). Set wait=true to poll until the video is ready (can take 11s to 6 min). Supports Veo 3.1 (best quality, 720p/1080p/4k, native audio), Veo 3.1 Fast, Veo 3, and Veo 2.`,
    inputSchema: GenerateVideoSchema,
    annotations: {
      readOnlyHint: false,
      destructiveHint: false,
      idempotentHint: false,
      openWorldHint: true,
    },
  },
  async (params) => {
    if (!GEMINI_API_KEY) {
      return {
        content: [{ type: "text", text: "Error: GEMINI_API_KEY environment variable is not set." }],
        isError: true,
      };
    }

    const parameters: Record<string, unknown> = {
      aspectRatio: params.aspect_ratio,
      resolution: params.resolution,
    };
    if (params.person_generation) parameters.personGeneration = params.person_generation;
    if (params.number_of_videos) parameters.numberOfVideos = params.number_of_videos;
    if (params.duration_seconds) parameters.durationSeconds = params.duration_seconds;
    if (params.include_audio !== undefined) parameters.includeAudio = params.include_audio;
    if (params.seed !== undefined) parameters.seed = params.seed;

    const body = {
      instances: [{ prompt: params.prompt }],
      parameters,
    };

    const result = await geminiRequest(
      `models/${params.model}:predictLongRunning`,
      "POST",
      body
    ) as { name: string; done?: boolean };

    const operationName = result.name;

    if (!params.wait) {
      return {
        content: [{
          type: "text",
          text: `Video generation submitted successfully.\n\nOperation: \`${operationName}\`\nModel: ${params.model}\nResolution: ${params.resolution}\nAspect Ratio: ${params.aspect_ratio}\n\nUse \`veo_check_operation\` with this operation name to check status and get the video URL when ready. Generation typically takes 11 seconds to 6 minutes.`,
        }],
      };
    }

    return await pollOperation(operationName);
  }
);

// --- Tool: Generate Video from Image ---

const ImageToVideoSchema = z.object({
  prompt: z.string().min(1).max(5000).describe("Text prompt describing the desired video motion/action"),
  image_url: z.string().url().describe("URL of the image to use as the starting frame"),
  model: z.string().default("veo-3.1-generate-preview").describe("Model ID"),
  aspect_ratio: z.enum(["16:9", "9:16"]).default("16:9").describe("Aspect ratio"),
  resolution: z.enum(["720p", "1080p", "4k"]).default("1080p").describe("Resolution"),
  include_audio: z.boolean().default(true).optional().describe("Include native audio"),
  wait: z.boolean().default(false).optional().describe("Poll until done"),
}).strict();

server.registerTool(
  "veo_image_to_video",
  {
    title: "Generate Video from Image",
    description: `Generate a video using an image as the starting frame. Fetches the image from the provided URL, converts it to base64, and uses it as the first frame for video generation. Combine with a text prompt to describe the desired motion.`,
    inputSchema: ImageToVideoSchema,
    annotations: {
      readOnlyHint: false,
      destructiveHint: false,
      idempotentHint: false,
      openWorldHint: true,
    },
  },
  async (params) => {
    if (!GEMINI_API_KEY) {
      return {
        content: [{ type: "text", text: "Error: GEMINI_API_KEY environment variable is not set." }],
        isError: true,
      };
    }

    const imageRes = await fetch(params.image_url);
    if (!imageRes.ok) {
      return {
        content: [{ type: "text", text: `Error: Could not fetch image from URL: ${imageRes.status}` }],
        isError: true,
      };
    }

    const imageBuffer = Buffer.from(await imageRes.arrayBuffer());
    const base64Image = imageBuffer.toString("base64");
    const contentType = imageRes.headers.get("content-type") || "image/png";

    const parameters: Record<string, unknown> = {
      aspectRatio: params.aspect_ratio,
      resolution: params.resolution,
    };
    if (params.include_audio !== undefined) parameters.includeAudio = params.include_audio;

    const body = {
      instances: [{
        prompt: params.prompt,
        image: {
          bytesBase64Encoded: base64Image,
          mimeType: contentType,
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
          text: `Image-to-video generation submitted.\n\nOperation: \`${result.name}\`\n\nUse \`veo_check_operation\` to check status.`,
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
  wait: z.boolean().default(false).optional().describe("Poll until done"),
}).strict();

server.registerTool(
  "veo_extend_video",
  {
    title: "Extend a Generated Video",
    description: `Extend a previously generated Veo video with new content. Takes the URI of a video generated by Veo and creates a new clip continuing from the last second of the previous clip. Allows building longer narratives by chaining generations. Video extension is limited to 720p resolution.`,
    inputSchema: ExtendVideoSchema,
    annotations: {
      readOnlyHint: false,
      destructiveHint: false,
      idempotentHint: false,
      openWorldHint: true,
    },
  },
  async (params) => {
    if (!GEMINI_API_KEY) {
      return {
        content: [{ type: "text", text: "Error: GEMINI_API_KEY environment variable is not set." }],
        isError: true,
      };
    }

    const body = {
      instances: [{
        prompt: params.prompt,
        video: {
          uri: params.video_uri,
          mimeType: "video/mp4",
        },
      }],
      parameters: {
        resolution: "720p",
      },
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

// --- Tool: Check Operation Status ---

const CheckOperationSchema = z.object({
  operation_name: z.string().min(1).describe("The operation name returned from a generate/extend call (e.g., 'operations/...')"),
}).strict();

server.registerTool(
  "veo_check_operation",
  {
    title: "Check Video Generation Status",
    description: `Check the status of a Veo video generation operation. Polls the Gemini API for the current status. Returns whether it is still processing or completed, along with video download URIs when ready. Videos are stored for 2 days on Google servers.`,
    inputSchema: CheckOperationSchema,
    annotations: {
      readOnlyHint: true,
      destructiveHint: false,
      idempotentHint: true,
      openWorldHint: true,
    },
  },
  async (params) => {
    if (!GEMINI_API_KEY) {
      return {
        content: [{ type: "text", text: "Error: GEMINI_API_KEY environment variable is not set." }],
        isError: true,
      };
    }

    const status = await geminiRequest(params.operation_name) as {
      done?: boolean;
      response?: {
        generateVideoResponse?: {
          generatedSamples?: Array<{ video?: { uri?: string } }>;
        };
      };
      error?: { message?: string };
      metadata?: unknown;
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
          text: `Status: **Processing**\n\nThe video is still being generated. Try checking again in 10-30 seconds.\n\nOperation: \`${params.operation_name}\``,
        }],
      };
    }

    const samples = status.response?.generateVideoResponse?.generatedSamples || [];
    if (samples.length === 0) {
      return {
        content: [{
          type: "text",
          text: "Status: **Completed** but no videos were returned. The content may have been filtered by safety checks.",
        }],
      };
    }

    const videoLines = samples.map((s, i) => {
      const uri = s.video?.uri || "No URI";
      return `Video ${i + 1}: ${uri}`;
    });

    return {
      content: [{
        type: "text",
        text: `Status: **Completed**\n\n${videoLines.join("\n")}\n\nVideos expire after 2 days. Download before they are removed. All videos include SynthID watermarks.`,
      }],
    };
  }
);

// --- Tool: Generate with Reference Images ---

const ReferenceImageSchema = z.object({
  prompt: z.string().min(1).max(5000).describe("Text prompt describing the video"),
  reference_image_urls: z.array(z.string().url()).min(1).max(3).describe("1-3 URLs of reference images to guide character/style consistency"),
  model: z.string().default("veo-3.1-generate-preview").describe("Model ID"),
  aspect_ratio: z.enum(["16:9", "9:16"]).default("16:9").describe("Aspect ratio"),
  resolution: z.enum(["720p", "1080p", "4k"]).default("1080p").describe("Resolution"),
  include_audio: z.boolean().default(true).optional().describe("Include native audio"),
  wait: z.boolean().default(false).optional().describe("Poll until done"),
}).strict();

server.registerTool(
  "veo_generate_with_references",
  {
    title: "Generate Video with Reference Images",
    description: `Generate a video guided by up to 3 reference images. Upload reference images to maintain character consistency, apply specific styles, or guide visual appearance. Supports Veo 3.1 only.`,
    inputSchema: ReferenceImageSchema,
    annotations: {
      readOnlyHint: false,
      destructiveHint: false,
      idempotentHint: false,
      openWorldHint: true,
    },
  },
  async (params) => {
    if (!GEMINI_API_KEY) {
      return {
        content: [{ type: "text", text: "Error: GEMINI_API_KEY environment variable is not set." }],
        isError: true,
      };
    }

    const referenceImages: Array<{ image: { bytesBase64Encoded: string; mimeType: string }; referenceType: string }> = [];

    for (const url of params.reference_image_urls) {
      const imgRes = await fetch(url);
      if (!imgRes.ok) {
        return {
          content: [{ type: "text", text: `Error: Could not fetch reference image from: ${url} (${imgRes.status})` }],
          isError: true,
        };
      }
      const buf = Buffer.from(await imgRes.arrayBuffer());
      const mime = imgRes.headers.get("content-type") || "image/png";
      referenceImages.push({
        image: { bytesBase64Encoded: buf.toString("base64"), mimeType: mime },
        referenceType: "asset",
      });
    }

    const parameters: Record<string, unknown> = {
      aspectRatio: params.aspect_ratio,
      resolution: params.resolution,
      referenceImages,
    };
    if (params.include_audio !== undefined) parameters.includeAudio = params.include_audio;

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
          text: `Video generation with ${params.reference_image_urls.length} reference image(s) submitted.\n\nOperation: \`${result.name}\`\n\nUse \`veo_check_operation\` to check status.`,
        }],
      };
    }

    return await pollOperation(result.name);
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

// --- SSE Transport (compatible with Claude.ai connectors) ---

const app = express();

const transports = new Map<string, SSEServerTransport>();

app.get("/sse", async (_req, res) => {
  const transport = new SSEServerTransport("/messages", res);
  transports.set(transport.sessionId, transport);

  res.on("close", () => {
    transports.delete(transport.sessionId);
  });

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
  res.json({ status: "ok", service: "veo-mcp-server", version: "1.0.0" });
});

app.listen(PORT, () => {
  console.log(`Veo MCP Server running on port ${PORT}`);
  console.log(`SSE endpoint: http://localhost:${PORT}/sse`);
  console.log(`API Key configured: ${GEMINI_API_KEY ? "Yes" : "No"}`);
});
