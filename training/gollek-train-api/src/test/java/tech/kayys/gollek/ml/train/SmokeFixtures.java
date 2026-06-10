package tech.kayys.gollek.ml.train;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

final class SmokeFixtures {
    private SmokeFixtures() {
    }

    static StableLabel stableLabel(String value) {
        return new StableLabel(value);
    }

    static List<Object> listWithNull(Object... values) {
        return java.util.Arrays.asList(values);
    }

    static Object nestedLists(Object leaf, int depth) {
        Object value = leaf;
        for (int index = 0; index < depth; index++) {
            value = List.of(value);
        }
        return value;
    }

    static Map<Integer, Integer> numberedMap(int size) {
        Map<Integer, Integer> values = new LinkedHashMap<>();
        for (int index = 0; index < size; index++) {
            values.put(index, index);
        }
        return values;
    }

    static List<Integer> numberedList(int size) {
        List<Integer> values = new ArrayList<>();
        for (int index = 0; index < size; index++) {
            values.add(index);
        }
        return values;
    }

    static CountingIterable countingIterable(int size) {
        return new CountingIterable(size);
    }

    record StableLabel(String value) {
        @Override public String toString() {
            return value;
        }
    }

    record CountingIterable(int size) implements Iterable<Integer> {
        @Override public java.util.Iterator<Integer> iterator() {
            return new java.util.Iterator<>() {
                private int index;

                @Override public boolean hasNext() {
                    return index < size;
                }

                @Override public Integer next() {
                    return index++;
                }
            };
        }
    }
}
