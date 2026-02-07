const std = @import("std");
const Allocator = std.mem.Allocator;

const Node = union(enum) {
    const Function = struct {
        arg: []const u8,
        body: *Node,
    };

    const App = struct {
        lhs: *Node,
        rhs: *Node,
    };

    variable: []const u8,
    function: Function,
    app: App,

    fn eql(a: Node, b: Node) bool {
        return switch (a) {
            .variable => |va| switch (b) {
                .variable => |vb| std.mem.eql(u8, va, vb),
                .function, .app => false,
            },
            .function => |fa| switch (b) {
                .function => |fb| std.mem.eql(u8, fa.arg, fb.arg) and fa.body.eql(fb.body.*),
                .variable, .app => false,
            },
            .app => |ca| switch (b) {
                .app => |cb| ca.lhs.eql(cb.lhs.*) and ca.rhs.eql(cb.rhs.*),
                .variable, .function => false,
            },
        };
    }

    fn variableOrPanic(e: Node) []const u8 {
        return switch (e) {
            .variable => |v| v,
            .function => @panic("error: expected variable, found function"),
            .app => @panic("error: expected variable, found application"),
        };
    }

    fn functionOrPanic(e: Node) Function {
        return switch (e) {
            .variable => @panic("error: expected function, found variable"),
            .function => |f| f,
            .app => @panic("error: expected function, found application"),
        };
    }

    /// Replace all occurences of `param` in `body` with `arg`.
    ///
    /// (\param.body) arg
    fn replace(
        arena: Allocator,
        param: []const u8,
        body: Node,
        arg: Node,
    ) Allocator.Error!Node {
        switch (body) {
            // (\param.v) arg
            .variable => {
                if (body.eql(.{ .variable = param })) return arg;
                return body;
            },
            // (\param.\inner_arg.inner_body) arg
            // We need to replace all occurences of `param` in `inner_arg`
            // and `inner_body` with `arg`.
            .function => |inner| {
                const new_inner_arg = new_inner_arg: {
                    const ret = try replace(arena, param, .{ .variable = inner.arg }, arg);
                    break :new_inner_arg ret.variableOrPanic();
                };

                const new_inner_body = try arena.create(Node);
                new_inner_body.* = try replace(arena, param, inner.body.*, arg);
                return Node{
                    .function = .{ .arg = new_inner_arg, .body = new_inner_body },
                };
            },
            // (\param.lhs rhs) arg
            // We need to replace all occurences of `param` in `lhs` and `rhs`
            // with `arg`.
            .app => |app| {
                const new_lhs = try arena.create(Node);
                const new_rhs = try arena.create(Node);
                new_lhs.* = try replace(arena, param, app.lhs.*, arg);
                new_rhs.* = try replace(arena, param, app.rhs.*, arg);
                return Node{
                    .app = .{ .lhs = new_lhs, .rhs = new_rhs },
                };
            }
        }
    }

    fn eval(self: Node, arena: Allocator) Allocator.Error!Node {
        switch (self) {
            .variable, .function => return self,
            .app => |app| {
                const f = f: {
                    const ret = try app.lhs.eval(arena);
                    break :f ret.functionOrPanic();
                };

                const replacement = try app.rhs.eval(arena);
                return try replace(arena, f.arg, f.body.*, replacement);
            },
        }
    }
};

const Token = union(enum) {
    ident: []const u8,
    eq,
    lambda,
    oparen,
    cparen,
    separator,
    space,
    end,
    illegal,

    fn match(b: u8) Token {
        return switch (b) {
            '=' => .eq,
            '\\' => .lambda,
            '(' => .oparen,
            ')' => .cparen,
            '.' => .separator,
            ' ' => .space,
            0 => .end,
            else => .{ .ident = "" },
        };
    }
};

const Lexer = struct {
    content: []const u8,
    cur: usize,

    fn init(content: []const u8) Lexer {
        return .{
            .content = content,
            .cur = 0,
        };
    }

    fn getAndAdvance(self: *Lexer) ?u8 {
        if (self.cur >= self.content.len) return null;

        const b = self.content[self.cur];
        self.cur += 1;

        return b;
    }

    fn next(self: *Lexer) Token {
        var start = self.cur;
        var b = self.getAndAdvance() orelse return .end;
        const tok = Token.match(b);
        switch (tok) {
            .eq, .lambda, .oparen, .cparen, .separator, .end, .illegal => return tok,
            // FIXME: Don't do recursion, chop instead whitespace.
            .space => return self.next(),
            .ident => {
                // We support overloaded identifiers with @"..." syntax,
                // therefore we need to check if we are overloading or not.
                if (b == '@') {
                    b = self.getAndAdvance() orelse return .illegal;
                    if (b != '"') return .illegal;

                    start = self.cur;
                    while (true) {
                        b = self.getAndAdvance() orelse return .illegal;
                        if (b == '"') break;
                    }
                } else {
                    while (true) {
                        b = self.getAndAdvance() orelse break;
                        if (Token.match(b) != .ident) break;
                    }
                }

                return .{ .ident = self.content[start..self.cur - 1] };
            },
        }
    }
};

pub fn main() !void {
    const content =
        \\ pair = \x.\y.\f.f x y
        \\ (@"first in pair" (pair 12 13))
    ;

    var l = Lexer.init(content);

    const exp = std.testing.expectEqual;
    const expStrings = std.testing.expectEqualStrings;

    try expStrings("pair", l.next().ident);
    try exp(.eq, l.next());
    try exp(.lambda, l.next());
    try expStrings("x", l.next().ident);
    try exp(.separator, l.next());
    try exp(.lambda, l.next());
    try expStrings("y", l.next().ident);
    try exp(.separator, l.next());
    try exp(.lambda, l.next());
    try expStrings("f", l.next().ident);
    try exp(.separator, l.next());
    try expStrings("f", l.next().ident);
    try expStrings("x", l.next().ident);
    try expStrings("y", l.next().ident);

    try exp(.oparen ,l.next());
    try expStrings("first in pair", l.next().ident);
    try exp(.oparen, l.next());
    try expStrings("pair", l.next().ident);
    try expStrings("12", l.next().ident);
    try expStrings("13", l.next().ident);
    try exp(.cparen, l.next());
    try exp(.cparen, l.next());

    try exp(Token.end, l.next());
}

test "lexer" {
    const content =
        \\ pair = \x.\y.\f.f x y
        \\ (@"first in pair" (pair 12 13))
    ;

    var l = Lexer.init(content);

    const exp = std.testing.expectEqual;
    const expStrings = std.testing.expectEqualStrings;

    try expStrings("pair", l.next().ident);
    try exp(.eq, l.next());
    try exp(.lambda, l.next());
    try expStrings("x", l.next().ident);
    try exp(.separator, l.next());
    try exp(.lambda, l.next());
    try expStrings("y", l.next().ident);
    try exp(.separator, l.next());
    try exp(.lambda, l.next());
    try expStrings("f", l.next().ident);
    try exp(.separator, l.next());
    try expStrings("f", l.next().ident);
    try expStrings("x", l.next().ident);
    try expStrings("y", l.next().ident);

    try exp(.oparen ,l.next());
    try expStrings("first in pair", l.next().ident);
    try exp(.oparen, l.next());
    try expStrings("pair", l.next().ident);
    try expStrings("12", l.next().ident);
    try expStrings("13", l.next().ident);
    try exp(.cparen, l.next());
    try exp(.cparen, l.next());

    try exp(Token.end, l.next());
}

test "eval variable" {
    var debug = std.heap.DebugAllocator(.{}).init;
    defer _ = debug.deinit();
    var arena_instance = std.heap.ArenaAllocator.init(debug.allocator());
    defer arena_instance.deinit();

    const expr = Node{
        .variable = "x y",
    };

    const ret = try expr.eval(arena_instance.allocator());
    const exp = Node{ .variable = "x y" };
    try std.testing.expect(ret.eql(exp));
}

test "eval function" {
    var debug = std.heap.DebugAllocator(.{}).init;
    defer _ = debug.deinit();
    var arena_instance = std.heap.ArenaAllocator.init(debug.allocator());
    defer arena_instance.deinit();

    const expr = Node{
        .function = .{
            .arg = "x",
            .body = @constCast(&Node{
                .variable = "x",
            }),
        },
    };

    const ret = try expr.eval(arena_instance.allocator());
    const exp = Node{
        .function = .{
            .arg = "x",
            .body = @constCast(&Node{
                .variable = "x",
            }),
        },
    };
    try std.testing.expect(ret.eql(exp));
}

test "eval simple application" {
    var debug = std.heap.DebugAllocator(.{}).init;
    defer _ = debug.deinit();
    var arena_instance = std.heap.ArenaAllocator.init(debug.allocator());
    defer arena_instance.deinit();

    const expr = Node{
        .app = .{
            .lhs = @constCast(&Node{
                .function = .{
                    .arg = "x",
                    .body = @constCast(&Node{
                        .variable = "x",
                    }),
                },
            }),
            .rhs = @constCast(&Node{ .variable = "y" }),
        },
    };

    const ret = try expr.eval(arena_instance.allocator());
    const exp = Node{ .variable = "y" };
    try std.testing.expect(ret.eql(exp));
}
