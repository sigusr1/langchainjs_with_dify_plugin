import { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import { BaseLanguageModelInput } from "@langchain/core/language_models/base";
import {
  AIMessage,
  AIMessageChunk,
  BaseMessage,
  ToolCall,
} from "@langchain/core/messages";
import {
  ChatGeneration,
  ChatGenerationChunk,
  ChatResult,
} from "@langchain/core/outputs";
import { Runnable } from "@langchain/core/runnables";
import { getEnvironmentVariable } from "@langchain/core/utils/env";
import { convertToOpenAITool } from "@langchain/core/utils/function_calling";

import {
  BaseChatModel,
  BindToolsInput,
  type BaseChatModelParams,
} from "@langchain/core/language_models/chat_models";

const TOOL_CALL_INSTRUCTION = `
You must follow OpenAI's function calling protocol strictly.

- If you need to call a tool, output ONLY a JSON object matching the OpenAI assistant message format with "tool_calls".
- The "arguments" field must be a STRING containing valid JSON (escaped properly).
- Generate a random unique "id" like "call_xxx" (e.g., "call_w9f3k2").
- NEVER add natural language, markdown, or extra fields.
- If no tool is needed, respond normally with natural language.

Example correct output of tool call:
{"role": "assistant", "tool_calls": [{"id": "call_a1b2c3", "type": "function", "function": {"name": "add", "arguments": "{"a": 5, "b": 3}"}}]}

Follow are tools available to you:
`;

export interface ChatDifyCallOptions extends BaseChatModelParams {
  user: string;
  apiKey?: string;
  baseUrl?: string;
  streaming?: boolean;
  tools?: BindToolsInput[];
}

export class ChatDify extends BaseChatModel<ChatDifyCallOptions> {
  // Used for tracing, replace with the same name as your class
  static lc_name() {
    return "ChatDify";
  }

  lc_serializable = true;

  /**
   * Replace with any secrets this class passes to `super`.
   * See {@link ../../langchain-cohere/src/chat_model.ts} for
   * an example.
   */
  get lc_secrets(): { [key: string]: string } | undefined {
    return {
      apiKey: "DIFY_API_KEY",
    };
  }

  get lc_aliases(): { [key: string]: string } | undefined {
    return {
      apiKey: "DIFY_API_KEY",
    };
  }

  // Replace
  _llmType() {
    return "dify";
  }

  apiKey?: string;
  user: string;
  baseUrl: string;
  streaming: boolean;
  conversation_id: "";

  constructor(fields: ChatDifyCallOptions) {
    super(fields);
    this.apiKey = fields.apiKey ?? getEnvironmentVariable("DIFY_API_KEY");
    if (!this.apiKey) {
      throw new Error(
        `Dify API key not found. Please set the DIFY_API_KEY environment variable or pass the key into "apiKey" field.`
      );
    }

    this.user = fields.user;
    this.baseUrl = fields.baseUrl ?? 'https://api.dify.ai/v1/chat-messages';
    this.streaming = fields.streaming ?? false;
  }

  override bindTools(
    tools: BindToolsInput[],
    kwargs?: Partial<this["ParsedCallOptions"]>
  ): Runnable<BaseLanguageModelInput, AIMessageChunk, ChatDifyCallOptions> {
    return this.withConfig({
      tools: tools.map((tool) => convertToOpenAITool(tool)),
      ...kwargs,
    });
  }

  private _allowStream(options: this["ParsedCallOptions"]): boolean {
    // Stream mode not support tool call, because we can't detect tool call until get all dify
    // outputs.
    if (this.streaming && !options?.tools?.length) {
      return true;
    } else {
      return false;
    }
  }

  private _assembleDifyPayload(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"]
  ) {
    const inputs: { [key: string]: any } = {};

    // Add tools
    const userDefinedToolCalls = options?.tools;
    if (userDefinedToolCalls?.length) {
      const openAITools = userDefinedToolCalls.map((tool) =>
        convertToOpenAITool(tool)
      );
      // console.log(`tool call info: ${JSON.stringify(openAITools)}`);
      inputs.tools = TOOL_CALL_INSTRUCTION + JSON.stringify(openAITools);
    }

    const queryString = [];
    for (const msg of messages) {
      if (msg.type == "ai") {
        queryString.push({
          role: "assistant",
          content: msg.content,
          tool_calls: (msg as AIMessage).tool_calls,
        });
        continue;
      }

      let type = msg.type;
      if (msg.type == "human") {
        type = "user";
      }
      queryString.push({ role: type, content: msg.content });
    }

    const mode = this._allowStream(options) ? "streaming" : "blocking";
    const payload = {
      response_mode: mode,
      user: this.user,
      inputs: inputs,
      query: JSON.stringify(queryString),
      // conversation_id: this.conversation_id
    };

    return payload;
  }

  private _parseToolCalls(str: string) {
    const toolCalls = [];
    const textJson = JSON.parse(str);
    for (const forgeToolCall of textJson?.tool_calls ?? []) {
      const parsedToolCall: ToolCall = {
        name: forgeToolCall.function.name,
        args: JSON.parse(forgeToolCall.function.arguments),
        type: "tool_call",
        id: forgeToolCall.id,
      };

      toolCalls.push(parsedToolCall);
      // console.log(`parsedToolCall:${JSON.stringify(parsedToolCall)}`)
    }

    return toolCalls;
  }

  private async _postToDify(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"]
  ) {
    const payload = this._assembleDifyPayload(messages, options);
    const response = await fetch(`${this.baseUrl}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify(payload),
    });

    return response;
  }

  override async _generate(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun
  ): Promise<ChatResult> {
    // stream response mode
    if (this._allowStream(options)) {
      const stream = this._streamResponseChunks(messages, options, runManager);
      let finalChunk: ChatGenerationChunk | undefined;
      for await (const chunk of stream) {
        if (finalChunk === undefined) {
          finalChunk = chunk;
        } else {
          finalChunk = finalChunk.concat(chunk);
        }
      }

      if (finalChunk === undefined) {
        throw new Error("No chunks returned from Dify API.");
      }

      return {
        generations: [
          {
            text: finalChunk.text,
            message: finalChunk.message,
          },
        ],
      };
    }

    // block response mode
    let text;
    let rspData;
    try {
      const response = await this._postToDify(messages, options);
      if (!response.ok) {
        console.log(`response:${JSON.stringify(response)}`);
        rspData = await response.json();
        throw new Error(
          `API error: ${response.status} ${response.statusText}}`
        );
      }

      rspData = await response.json();
      // console.log(`rspData:${JSON.stringify(rspData)}`)
      text = rspData.answer;
      this.conversation_id = rspData.conversation_id || "";
      const toolCalls = this._parseToolCalls(text);
      if (toolCalls.length > 0) {
        const aiMessage = new AIMessage({
          tool_calls: toolCalls,
          id: rspData.id,
          content: "",
        });

        const generation: ChatGeneration = {
          text: "",
          message: aiMessage,
        };

        return { generations: [generation] };
      }
    } catch (e: any) {
      if (!rspData || !rspData.answer) {
        console.error(e);
        throw e;
      }
      // If parse meet error, rspData not null, follow down as normal AIMessage.
    }

    const message = new AIMessage(text);

    return {
      generations: [
        {
          text,
          message,
        },
      ],
    };
  }

  /**
   * Implement to support streaming.
   * Should yield chunks iteratively.
   */
  override async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    let jsonStr;
    try {
      const response = await this._postToDify(messages, options);
      if (!response.body) throw new Error("No response body");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        // Parse dify output, refer to https://github.com/fatwang2/dify2openai
        buffer += decoder.decode(value, { stream: true });
        // console.log(`buffer:\n ${JSON.stringify(buffer)}`)
        const lines = buffer.split("\n");
        for (const line of lines) {
          if (line.trim() === "") continue;
          if (line.startsWith(":")) continue;

          // one line one json
          jsonStr = line;
          if (line.startsWith("data: ")) {
            jsonStr = line.slice(6); // Remove "data: "
          }

          try {
            const chunk = JSON.parse(jsonStr); // Maybe not a complete json
            this.conversation_id = chunk.conversation_id || "";
            const content = chunk.answer || "";
            if (content) {
              yield new ChatGenerationChunk({
                text: content,
                message: new AIMessageChunk(content),
              });
            }
          } catch {
            continue;
          }
        }
        buffer = lines.length > 0 ? lines[lines.length - 1] : "";
      }
    } catch (e) {
      console.warn(`jsonStr is :${jsonStr}`);
      throw e;
    }
  }
}
