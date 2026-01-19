# @langchain/dify

This package contains the LangChain.js integrations for dify.

## Build

In your project's root directory, execute commands:

```bash
pnpm install
```

```bash
pnpm build
```

## Run Demo

### 1. Create chatflow on Dify

Create a basic chatflow on Dify:
![chatflow](https://github.com/sigusr1/langchainjs_with_dify_plugin/blob/main/libs/providers/langchain-dify/assets/chatflow.jpg)

Please pay attention to parameter `tools`, which is filled by langchain/dify and used by llm:
![tools](https://github.com/sigusr1/langchainjs_with_dify_plugin/blob/main/libs/providers/langchain-dify/assets/tool_param.jpg)


Then set `tools` and `query` to llm: 

![tools_and_query](https://github.com/sigusr1/langchainjs_with_dify_plugin/blob/main/libs/providers/langchain-dify/assets/tools_and_query.jpg)


### 2. Run Demo

In your project's example directory, execute commands:

Demo show the basic usage of `ChatDify`:
```bash
pnpm run start src/provider/dify/chat.ts
```

Demo show tool call of `ChatDify`:
```bash
pnpm run start src/provider/dify/llmWithTool.ts
```

Demo show `ChatDify` collaborate with `ReactAgent`:
```bash
pnpm run start src/provider/dify/agentWithTools.ts
```
