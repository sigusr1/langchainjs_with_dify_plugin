import { DynamicStructuredTool } from "@langchain/core/tools";
import { z } from "zod";
import { ChatDify } from "@langchain/dify";
import {
  BaseMessage,
  HumanMessage,
  ToolMessage,
} from "@langchain/core/messages";

async function getCurrentWeather(
  location: string,
  unit: "celsius" | "fahrenheit" = "celsius"
) {
  const mockData = {
    location,
    temperature: Math.floor(Math.random() * 30) + 10,
    unit,
    description: "Sunny",
  };

  return JSON.stringify(mockData);
}

export const weatherTool = new DynamicStructuredTool({
  name: "fetch_weather",
  description:
    "Get the current weather in a given location, if not provide temperature unit, use default",
  schema: z.object({
    location: z.string().describe("The city and state, e.g. San Francisco, CA"),
    unit: z.enum(["celsius", "fahrenheit"]).default("celsius").optional(),
  }),
  func: async ({ location, unit }) => {
    return await getCurrentWeather(location, unit as any);
  },
});

export const llm = new ChatDify({
  user: "huohuohuo",
  apiKey: "app-xxxxxxxx",
}).bindTools([weatherTool]);

export async function main(userInput: string) {
  const messages: BaseMessage[] = [
    new HumanMessage(userInput),
  ];

  while (true) {
    const response = await llm.invoke(messages);
    if (!response.tool_calls || response.tool_calls.length === 0) {
      return response.content;
    }

    for (const toolCall of response.tool_calls) {
      console.log(
        `🔧 Calling tool: ${toolCall.name} with args ${typeof toolCall.args}:`,
        toolCall.args
      );

      let result: string;
      if (toolCall.name === "fetch_weather") {
        result = await weatherTool.invoke(
          toolCall.args as z.infer<typeof weatherTool.schema>
        );
      } else {
        result = `Unknown tool: ${toolCall.name}`;
      }
      messages.push(
        new ToolMessage({
          content: result,
          tool_call_id: toolCall.id!,
          name: toolCall.name,
        })
      );
    }
  }
}

console.log("🤖 User: What's the weather like in Boston and hangzhou?");
const answer = await main(
  "What's the weather like in Boston and hangzhou?"
);
console.log("💬 AI:", answer);
