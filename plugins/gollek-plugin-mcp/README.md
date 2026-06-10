
# Complete MCP Protocol Implementation

## 📦 Module 4: MCP Client & Protocol

## External `mcpServers` JSON Input

This plugin can load server definitions from standard MCP JSON input, not only from
`gollek.mcp.servers` YAML blocks.

Supported input keys:
- `gollek.mcp.mcp-servers-json` (inline JSON string; useful for CLI `-D`)
- `gollek.mcp.mcp-servers-json-file` (path to JSON file)

Accepted JSON shape:

```json
{
  "mcpServers": {
    "image-downloader": {
      "command": "node",
      "args": ["/path/to/mcp-image-downloader/build/index.js"]
    }
  }
}
```

Notes:
- `mcpServers.<name>.transport` defaults to `stdio`
- `name` is optional; when omitted, the object key is used as the server name
- If both YAML and JSON define the same server name, JSON overrides YAML

### Project Structure

```
inference-provider-mcp/
├── pom.xml
└── src/main/java/tech/kayys/gollek/mcp/
    ├── client/
    │   ├── MCPClient.java
    │   ├── MCPClientConfig.java
    │   ├── MCPConnection.java
    │   ├── MCPTransport.java
    │   ├── StdioTransport.java
    │   ├── HttpTransport.java
    │   └── WebSocketTransport.java
    ├── protocol/
    │   ├── MCPMessage.java
    │   ├── MCPRequest.java
    │   ├── MCPResponse.java
    │   ├── MCPNotification.java
    │   ├── MCPError.java
    │   └── JsonRpcMessage.java
    ├── server/
    │   ├── MCPServerRegistry.java
    │   ├── MCPServerDescriptor.java
    │   └── MCPServerConnection.java
    ├── tools/
    │   ├── MCPTool.java
    │   ├── MCPToolExecutor.java
    │   ├── MCPToolRegistry.java
    │   └── MCPToolResult.java
    ├── resources/
    │   ├── MCPResourceProvider.java
    │   ├── MCPResource.java
    │   └── MCPResourceContent.java
    ├── prompts/
    │   ├── MCPPromptProvider.java
    │   └── MCPPrompt.java
    ├── provider/
    │   ├── MCPProvider.java
    │   ├── MCPProviderFactory.java
    │   └── RemoteMCPProvider.java
    └── memory/
        ├── MCPMemoryProvider.java
        └── MCPMemoryStore.java
```
