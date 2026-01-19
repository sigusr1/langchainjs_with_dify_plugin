import { HumanMessage } from "@langchain/core/messages";
import { ChatDify } from "@langchain/dify";
const model = new ChatDify({
  user: "huohuohuo", // arbitrarily string
  apiKey: "app-xxxxxxxx", // API key
  // baseUrl: "https://api.dify.ai/v1/chat-messages", // Or your self host url.
  streaming: true, // default is false
});

let messages = [
  new HumanMessage("Hello! What is 2+5?"),
];

let stream = await model.stream(messages);
for await (const chunk of stream) {
  console.log(chunk.content);
}

messages = [new HumanMessage("Why the sky is blue?")];
stream = await model.stream(messages);
for await (const chunk of stream) {
  console.log(chunk.content);
}
