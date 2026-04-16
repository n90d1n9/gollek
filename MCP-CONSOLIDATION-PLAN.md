# MCP Module Consolidation Plan

## Problem
Two parallel MCP implementations exist:
1. `gollek/core/gollek-mcp` - Orphaned, cannot build, no dependents
2. `gollek/plugins/common/gollek-plugin-mcp` - Buildable, actively used, more complete

## Solution
Consolidate into `gollek-plugin-mcp` and remove `gollek/core/gollek-mcp`.

## Migration Steps

### 1. Move MCP-2025-11-25 Protocol Types
**From:** `gollek/core/gollek-mcp/src/main/java/tech/kayys/gollek/agent/mcp/McpProtocol.java`
**To:** `gollek/plugins/common/gollek-plugin-mcp/src/main/java/tech/kayys/gollek/mcp/protocol/McpProtocol.java`

The protocol types should be in a shared location since they're pure DTOs.

### 2. Move Transport Implementations
**From:** 
- `gollek/core/gollek-mcp/src/main/java/tech/kayys/gollek/agent/mcp/McpSseClient.java`
- `gollek/core/gollek-mcp/src/main/java/tech/kayys/gollek/agent/mcp/McpStdioClient.java`

**To:**
- `gollek/plugins/common/gollek-plugin-mcp/src/main/java/tech/kayys/gollek/mcp/client/SseTransport.java` (already exists)
- `gollek/plugins/common/gollek-plugin-mcp/src/main/java/tech/kayys/gollek/mcp/client/StdioTransport.java` (already exists)

The plugin module already has these transports!

### 3. Move McpClient SPI Interface
**From:** `gollek/core/gollek-mcp/src/main/java/tech/kayys/gollek/agent/mcp/McpClient.java`
**To:** `gollek/plugins/common/gollek-plugin-mcp/src/main/java/tech/kayys/gollek/mcp/McpClient.java`

Rename package from `tech.kayys.gollek.agent.mcp` to `tech.kayys.gollek.mcp`.

### 4. Rewrite McpServerRegistry & McpSkillAdapter
These are the only unique pieces in the core module:
- `McpServerRegistry` - Connects to MCP servers and registers tools as skills
- `McpSkillAdapter` - Bridges MCP tools to AgentSkill SPI

**Option A:** Move to plugin module
- Add as optional CDI beans in plugin
- Only activate when agent-skill system is available

**Option B:** Move to wayang-gollek module
- Keep in agent layer where they belong
- Use plugin's `MCPClient` instead of core's `McpClient`

**Recommendation:** Option B - these are agent-level concerns, not MCP plugin concerns.

### 5. Update McpSkillAdapter
Change from using core module's `McpClient` to plugin's `MCPClient`:

```java
// OLD (uses core module)
import tech.kayys.gollek.agent.mcp.McpClient;
import tech.kayys.gollek.agent.mcp.McpProtocol.McpTool;

// NEW (uses plugin module)
import tech.kayys.gollek.mcp.client.MCPClient;
import tech.kayys.gollek.mcp.client.MCPConnection;
import tech.kayys.gollek.mcp.dto.MCPTool;
```

### 6. Remove Orphaned Module
```bash
# Remove the entire core module
rm -rf gollek/core/gollek-mcp

# Remove loose McpSkillAdapter.java
rm -f wayang-gollek/agent/McpSkillAdapter.java
```

### 7. Update References
- Update any remaining references to old package names
- Ensure all imports point to `tech.kayys.gollek.mcp.*`
- Update documentation

## Benefits

1. **Single Source of Truth** - One MCP implementation, not two
2. **Buildable** - No more orphaned code
3. **Maintainable** - No duplication of protocol types, transports, etc.
4. **Cleaner Architecture** - Plugin owns MCP protocol, agent layer owns skill adaptation
5. **MCP 2025-11-25 Compliant** - All new features preserved in consolidated module

## Architecture After Consolidation

```
gollek/plugins/common/gollek-plugin-mcp/
├── client/
│   ├── MCPClient.java          # Main CDI bean, manages connections
│   ├── MCPTransport.java       # Transport SPI
│   ├── SseTransport.java       # HTTP/SSE implementation
│   ├── StdioTransport.java     # stdio implementation
│   └── MCPClientConfig.java    # Configuration
├── dto/                        # All MCP protocol types
│   ├── McpProtocol.java        # Protocol types (moved from core)
│   ├── MCPTool.java
│   ├── MCPResource.java
│   ├── MCPPrompt.java
│   ├── MCPContentBlock.java
│   ├── RootInfo.java
│   └── ...
├── provider/                   # LLMProvider integration
│   ├── MCPProvider.java
│   └── ...
├── tool/                       # Tool execution
├── resource/                   # Resource management
├── prompt/                     # Prompt handling
└── registry/                   # Server registry (McpRegistryEngine)

wayang-gollek/agent/
└── mcp/
    ├── McpServerRegistry.java  # Moved from core, uses plugin's MCPClient
    └── McpSkillAdapter.java    # Moved from core, bridges to AgentSkill
```

## Risk Assessment

**Low Risk:**
- Core module has no dependents
- Core module cannot build
- All functionality exists in plugin module
- Only McpSkillAdapter needs rewriting (simple adapter class)

**Mitigation:**
- Keep backup of core module in git history
- Test McpSkillAdapter rewrite thoroughly
- Verify all MCP 2025-11-25 features preserved

## Timeline

1. Move protocol types to plugin (15 min)
2. Verify transports exist in plugin (already done ✓)
3. Move McpClient interface to plugin (10 min)
4. Rewrite McpServerRegistry & McpSkillAdapter (30 min)
5. Remove core module (5 min)
6. Test build (10 min)
7. Update documentation (10 min)

**Total: ~80 minutes**
