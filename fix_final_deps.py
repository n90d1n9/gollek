import os
import glob

# Remove gollek-engine from all provider build files
provider_files = glob.glob('./provider/*/build.gradle.kts') + glob.glob('./plugins/*/build.gradle.kts')
for pf in provider_files:
    if os.path.exists(pf):
        with open(pf, 'r') as f:
            content = f.read()
        changed = False
        if 'gollek-engine' in content:
            content = content.replace('implementation(group = "tech.kayys.gollek", name = "gollek-engine")', '')
            changed = True
        if 'gollek-model-core' in content:
            content = content.replace('implementation(group = "tech.kayys.gollek", name = "gollek-model-core")', '')
            changed = True
        if changed:
            with open(pf, 'w') as f:
                f.write(content)

# Fix RequestValidationPlugin.java missing return statement
req_path = './plugins/gollek-plugin-content-safety/src/main/java/tech/kayys/gollek/plugin/RequestValidationPlugin.java'
if os.path.exists(req_path):
    with open(req_path, 'r') as f:
        content = f.read()

    # find the shutdown method block and ensure it returns
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'public io.smallrye.mutiny.Uni<Void> shutdown()' in line:
            # check if it already returns
            if 'return' not in lines[i+1] and 'return' not in lines[i+2]:
                lines.insert(i+2, '        return io.smallrye.mutiny.Uni.createFrom().voidItem();')
                break
    with open(req_path, 'w') as f:
        f.write('\n'.join(lines))
