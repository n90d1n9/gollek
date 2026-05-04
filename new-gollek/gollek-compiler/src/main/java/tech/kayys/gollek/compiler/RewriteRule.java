package tech.kayys.gollek.compiler;

interface RewriteRule {
boolean match(List<GOp> ops, int index);
List<GOp> rewrite(List<GOp> ops, int index);
}