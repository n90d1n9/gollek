package tech.kayys.gollek.jupyter;

import org.dflib.jjava.jupyter.kernel.display.DisplayData;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

final class NotebookMagicDispatcher {

    @FunctionalInterface
    interface Handler {
        DisplayData handle(String argument) throws Exception;
    }

    private record PrefixRoute(String magic, Handler handler) {}

    private final Map<String, Handler> exactRoutes = new LinkedHashMap<>();
    private final List<PrefixRoute> prefixRoutes = new ArrayList<>();

    NotebookMagicDispatcher exact(String magic, Handler handler) {
        exactRoutes.put(magic, handler);
        return this;
    }

    NotebookMagicDispatcher prefix(String magic, Handler handler) {
        prefixRoutes.add(new PrefixRoute(magic, handler));
        return this;
    }

    DisplayData evaluate(String code) throws Exception {
        String trimmed = code == null ? "" : code.trim();
        if (!trimmed.startsWith("%")) {
            return null;
        }
        Handler exact = exactRoutes.get(trimmed);
        if (exact != null) {
            return exact.handle("");
        }
        DisplayData usage = NotebookMagicCatalog.renderUsage(trimmed);
        if (usage != null) {
            return usage;
        }
        for (PrefixRoute route : prefixRoutes) {
            String prefix = route.magic() + " ";
            if (trimmed.startsWith(prefix)) {
                return route.handler().handle(trimmed.substring(route.magic().length()).trim());
            }
        }
        return NotebookMagicCatalog.unknownMagic(trimmed);
    }
}
