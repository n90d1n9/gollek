import os

projects = [
    './optimization/gollek-plugin-perfmode/build.gradle.kts',
    './optimization/gollek-plugin-wait-scheduler/build.gradle.kts',
    './optimization/gollek-plugin-evicpress/build.gradle.kts',
    './optimization/gollek-plugin-prompt-cache/build.gradle.kts'
]

for p in projects:
    if os.path.exists(p):
        with open(p, 'r') as f:
            content = f.read()
        content = content.replace('implementation(group = "tech.kayys.gollek", name = "gollek-engine")', '')
        with open(p, 'w') as f:
            f.write(content)

req_plugin = './plugins/gollek-plugin-content-safety/src/main/java/tech/kayys/gollek/plugin/RequestValidationPlugin.java'
if os.path.exists(req_plugin):
    with open(req_plugin, 'r') as f:
        content = f.read()
    content = content.replace('public void initialize', 'public io.smallrye.mutiny.Uni<Void> initialize')
    content = content.replace('public void shutdown', 'public io.smallrye.mutiny.Uni<Void> shutdown')
    content = content.replace('LOG.info("RequestValidationPlugin initialized");\n    }', 'LOG.info("RequestValidationPlugin initialized");\n        return io.smallrye.mutiny.Uni.createFrom().voidItem();\n    }')
    content = content.replace('LOG.info("RequestValidationPlugin shutdown");\n    }', 'LOG.info("RequestValidationPlugin shutdown");\n        return io.smallrye.mutiny.Uni.createFrom().voidItem();\n    }')
    with open(req_plugin, 'w') as f:
        f.write(content)

