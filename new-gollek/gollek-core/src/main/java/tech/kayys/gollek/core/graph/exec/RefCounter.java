package tech.kayys.gollek.core.graph.exec;

import tech.kayys.gollek.core.graph.*;
import java.util.*;

import org.w3c.dom.Node;

public final class RefCounter {
    public static Map<Node, Integer> count(Node root) {
        Map<Node, Integer> ref = new HashMap<>();
        traverse(root, ref);
        return ref;
    }

    private static void traverse(Node node, Map<Node, Integer> ref) {
        if (!(node instanceof HasInputs hi))
            return;
        for (Node in : hi.inputs()) {
            ref.put(in, ref.getOrDefault(in, 0) + 1);

            traverse(in, ref);
        }
    }
}
