import { z } from "zod";
import { createAgent, tool } from "langchain";
import { InMemoryStore, LangGraphRunnableConfig } from "@langchain/langgraph";
import { ChatDify } from "@langchain/dify";

const getUserInfo = tool(
  async ({ userId }, config: LangGraphRunnableConfig) => {
    console.log(`call tool getUserInfo: ${userId}`);
    return config.store?.get(["users"], userId);
  },
  {
    name: "get_user_info",
    description: "Look up user info.",
    schema: z.object({
      userId: z.string(),
    }),
  }
);

const saveUserInfo = tool(
  async ({ userId, name, age, email }, config: LangGraphRunnableConfig) => {
    await config.store?.put(["users"], userId, { name, age, email });
    console.log(`call tool saveUserInfo: ${userId}`);

    return "Successfully saved user info.";
  },
  {
    name: "save_user_info",
    description: "Save user info.",
    schema: z.object({
      userId: z.string(),
      name: z.string(),
      age: z.number(),
      email: z.string(),
    }),
  }
);

const agent = createAgent({
  model: new ChatDify({
    apiKey: "app-xxxxxx",
    baseUrl: "https://api.dify.ai/v1/chat-messages",
    streaming: false,
    user: "huohuohuo",
  }),
  tools: [getUserInfo, saveUserInfo],
  store: new InMemoryStore(),
});

// First session: save user info
let result = await agent.invoke({
  messages:
    "Save the following user: userid: abc123, name: Foo, age: 25, email: foo@langchain.dev",
});

console.log(result.messages.at(-1)?.content);

// Second session: get user info
result = await agent.invoke({
  messages: "Get user info for user with id 'abc123'",
});
// Here is the user info for user with ID "abc123":
// - Name: Foo
// - Age: 25
// - Email: foo@langchain.dev
console.log(result.messages.at(-1)?.content);
