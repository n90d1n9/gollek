package tech.kayys.gollek.compiler;

public final class MemorySlot {
private final int id;
public MemorySlot(int id) {
this.id = id;
}
public int id() { return id; }
}