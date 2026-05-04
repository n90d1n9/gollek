package tech.kayys.gollek.compiler;

import tech.kayys.gollek.ir.*;
import java.util.*;
public final class MemoryPlanner {
public Map<GValueId, MemorySlot> plan(GGraph graph) {
Map<GValueId, Integer> lastUse = computeLastUse(graph);
Map<GValueId, MemorySlot> allocation = new HashMap<>();
List<MemorySlot> freeSlots = new ArrayList<>();
int nextSlot = 0;
for (int i = 0; i < graph.ops().size(); i++) {
GOp op = graph.ops().get(i);
for (GValueId out : op.outputs()) {
MemorySlot slot;
if (!freeSlots.isEmpty()) {
slot = freeSlots.remove(0);
} else {
slot = new MemorySlot(nextSlot++);
}
allocation.put(out, slot);
}
for (GValueRef in : op.inputs()) {
if (lastUse.get(in.id()) == i) {
freeSlots.add(allocation.get(in.id()));
}
}
}
return allocation;
}
private Map<GValueId, Integer> computeLastUse(GGraph graph) {
Map<GValueId, Integer> map = new HashMap<>();
List<GOp> ops = graph.ops();
for (int i = 0; i < ops.size(); i++) {
for (GValueRef in : ops.get(i).inputs()) {
map.put(in.id(), i);
}
}
return map;
}
}