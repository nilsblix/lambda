const std = @import("std");
const Allocator = std.mem.Allocator;
const Writer = std.Io.Writer;

const Ast = struct {
    root: *Node,

    fn init(alloc: Allocator, root: Node) Allocator.Error!Ast {
        const r = try alloc.create(Node);
        r.* = root;
        return Ast{
            .root = r,
        };
    }

    fn deinit(self: *Ast, alloc: Allocator) void {
        self.root.deinit(alloc);
        alloc.destroy(self.root);
    }

    pub fn format(self: Ast, w: *Writer) Writer.Error!void {
        try w.print("{f}", .{self.root});
    }

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

        fn deinit(e: *Node, alloc: Allocator) void {
            switch (e.*) {
                .variable => {},
                .function => |f| {
                    f.body.deinit(alloc);
                    alloc.destroy(f.body);
                },
                .app => |app| {
                    app.lhs.deinit(alloc);
                    app.rhs.deinit(alloc);
                    alloc.destroy(app.lhs);
                    alloc.destroy(app.rhs);
                },
            }
        }

        pub fn format(e: Node, w: *Writer) Writer.Error!void {
            switch (e) {
                .variable => |i| try w.print("{s}", .{i}),
                .function => |f| try w.print("\\{s}.{f}", .{ f.arg, f.body.* }),
                .app => |app| try w.print("({f} {f})", .{ app.lhs.*, app.rhs.* }),
            }
        }

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

        /// `body` needs to be freed after calling this, as all heap allocated
        /// members get duped in this function.
        ///
        /// All members of the returned Node are heap allocated, while the top
        /// level Node is copied, thus on the stack.
        ///
        /// (\param.body) arg
        fn replace(
            alloc: Allocator,
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
                        const ret = try replace(alloc, param, .{ .variable = inner.arg }, arg);
                        break :new_inner_arg ret.variableOrPanic();
                    };

                    const new_inner_body = try alloc.create(Node);
                    new_inner_body.* = try replace(alloc, param, inner.body.*, arg);
                    return Node{
                        .function = .{ .arg = new_inner_arg, .body = new_inner_body },
                    };
                },
                // (\param.lhs rhs) arg
                // We need to replace all occurences of `param` in `lhs` and `rhs`
                // with `arg`.
                .app => |app| {
                    const new_lhs = try alloc.create(Node);
                    const new_rhs = try alloc.create(Node);
                    new_lhs.* = try replace(alloc, param, app.lhs.*, arg);
                    new_rhs.* = try replace(alloc, param, app.rhs.*, arg);
                    return Node{
                        .app = .{ .lhs = new_lhs, .rhs = new_rhs },
                    };
                },
            }
        }

        fn eval(self: *const Node, alloc: Allocator) Allocator.Error!*Node {
            switch (self.*) {
                // .variable, .function => return @constCast(self),
                .variable, .function => {
                    const buf = try alloc.dupe(Node, @ptrCast(self));
                    return &buf[0];
                },
                .app => |app| {
                    const f = f: {
                        const ret = try app.lhs.eval(alloc);
                        break :f ret.functionOrPanic();
                    };

                    const replacement = try app.rhs.eval(alloc);
                    defer {
                        replacement.deinit(alloc);
                        alloc.destroy(replacement);
                    }

                    const ret = try alloc.create(Node);
                    ret.* = try replace(alloc, f.arg, f.body.*, replacement.*);
                    return ret;
                },
            }
        }
    };
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
            ' ', '\n', '\r', '\t' => .space,
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

    fn byteAndAdvance(self: *Lexer) ?u8 {
        if (self.cur >= self.content.len) return null;

        const b = self.content[self.cur];
        self.cur += 1;

        return b;
    }

    fn peekedByte(self: *Lexer) ?u8 {
        const cur = self.cur;
        const b = self.byteAndAdvance();
        self.cur = cur;
        return b;
    }

    fn next(self: *Lexer) Token {
        var start = self.cur;
        var b = self.byteAndAdvance() orelse return .end;
        const tok = Token.match(b);
        switch (tok) {
            .eq, .lambda, .oparen, .cparen, .separator, .end, .illegal => return tok,
            // FIXME: Don't do recursion, chop instead whitespace.
            .space => return self.next(),
            .ident => {
                // We support overloaded identifiers with @"..." syntax,
                // therefore we need to check if we are overloading or not.
                if (b == '@') {
                    b = self.byteAndAdvance() orelse return .illegal;
                    if (b != '"') return .illegal;

                    start = self.cur;
                    while (true) {
                        b = self.byteAndAdvance() orelse return .illegal;
                        if (b == '"') break;
                    }
                    return .{ .ident = self.content[start .. self.cur - 1] };
                } else {
                    while (true) {
                        b = self.peekedByte() orelse break;
                        if (Token.match(b) != .ident) break;
                        _ = self.byteAndAdvance();
                    }
                    return .{ .ident = self.content[start..self.cur] };
                }
            },
        }
    }
};

fn parseExpressionWithLexer(alloc: Allocator, l: *Lexer) Allocator.Error!?Ast {
    switch (l.next()) {
        .ident => |i| return try Ast.init(alloc, .{ .variable = i }),
        .lambda => {
            const arg = arg: {
                const ret = l.next();
                break :arg switch (ret) {
                    .ident => |i| i,
                    .eq, .lambda, .oparen, .cparen, .separator, .space, .end, .illegal => {
                        std.log.err("expected `ident`, found `{}`", .{ret});
                        return null;
                    },
                };
            };

            const next = l.next();
            if (next != .separator) {
                std.log.err("expected separator, found `{}`", .{next});
                return null;
            }

            const body = try parseExpressionWithLexer(alloc, l) orelse return null;

            return try Ast.init(alloc, .{ .function = .{ .arg = arg, .body = body.root } });
        },
        .oparen => {
            const f = f: {
                const ret = try parseExpressionWithLexer(alloc, l) orelse return null;
                if (ret.root.* != .function) {
                    std.log.err("expected function expression, found `{f}`", .{ret.root.*});
                    return null;
                }
                break :f ret;
            };

            const arg = try parseExpressionWithLexer(alloc, l) orelse return null;

            const end = l.byteAndAdvance();
            if (end == null or Token.match(end.?) != .cparen) {
                const c = end orelse 0;
                std.log.err("expected ')' to close application, found `{c}`", .{c});
                return null;
            }

            return try Ast.init(alloc, .{ .app = .{ .lhs = f.root, .rhs = arg.root } });
        },
        .eq, .cparen, .separator, .space, .end, .illegal => return null,
    }
}

const Result = struct { Ast, usize };

/// Returns the Ast and the position advanced to in the `input` slice. Position
/// is useful for evaluating multiple expressions from the same input.
fn parseExpression(alloc: Allocator, input: []const u8) Allocator.Error!?Result {
    var l = Lexer.init(input);
    const ast = try parseExpressionWithLexer(alloc, &l) orelse return null;
    return .{ ast, l.cur + 1 };
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}).init;
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const f = std.fs.File.stdout();
    defer f.close();

    var reader_buf: [4096]u8 = undefined;
    var reader = f.reader(&reader_buf);

    while (true) {
        std.debug.print("@> ", .{});
        const input = try reader.interface.takeDelimiter('\n') orelse continue;
        if (std.mem.eql(u8, input, ":quit")) break;

        var ast, _ = try parseExpression(alloc, input) orelse continue;
        defer ast.deinit(alloc);

        const res = try ast.root.eval(alloc);
        // defer {
        //     res.deinit(alloc);
        //     alloc.destroy(res);
        // }

        std.debug.print("{f}\n", .{res});
    }
}

test "parse expression and deinit" {
    var debug = std.heap.DebugAllocator(.{}).init;
    defer _ = debug.deinit();
    const gpa = debug.allocator();

    {
        const input =
            \\ var
        ;
        var ast, _ = try parseExpression(gpa, input) orelse return error.UnexpectedResult;
        defer ast.deinit(gpa);
        const exp = Ast.Node{ .variable = "var" };
        try std.testing.expect(ast.root.eql(exp));
    }

    {
        const input =
            \\ \x.some_return
        ;

        var ast, _ = try parseExpression(gpa, input) orelse return error.UnexpectedResult;
        defer ast.deinit(gpa);
        const exp = Ast.Node{
            .function = .{
                .arg = "x",
                .body = @constCast(&Ast.Node{
                    .variable = "some_return",
                }),
            },
        };
        try std.testing.expect(ast.root.eql(exp));
    }

    {
        const input =
            \\ (\x.some_return some_arg)
        ;
        var ast, _ = try parseExpression(gpa, input) orelse return error.UnexpectedResult;
        defer ast.deinit(gpa);
        const exp = Ast.Node{
            .app = .{
                .lhs = @constCast(&Ast.Node{
                    .function = .{
                        .arg = "x",
                        .body = @constCast(&Ast.Node{
                            .variable = "some_return",
                        }),
                    },
                }),
                .rhs = @constCast(&Ast.Node{
                    .variable = "some_arg",
                }),
            },
        };
        try std.testing.expect(ast.root.eql(exp));
    }

    {
        const input =
            \\ \x.some_return \wow.\some.more
        ;
        var ast, const pos = try parseExpression(gpa, input) orelse return error.UnexpectedResult;
        var exp = Ast.Node{
            .function = .{
                .arg = "x",
                .body = @constCast(&Ast.Node{
                    .variable = "some_return",
                }),
            },
        };
        try std.testing.expect(ast.root.eql(exp));
        ast.deinit(gpa);

        ast, _ = try parseExpression(gpa, input[pos..]) orelse return error.UnexpectedResult;
        defer ast.deinit(gpa);
        exp = Ast.Node{
            .function = .{
                .arg = "wow",
                .body = @constCast(&Ast.Node{
                    .function = .{
                        .arg = "some",
                        .body = @constCast(&Ast.Node{
                            .variable = "more",
                        }),
                    },
                }),
            },
        };
        try std.testing.expect(ast.root.eql(exp));
    }
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

    try exp(.oparen, l.next());
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

    const expr = Ast.Node{
        .variable = "x y",
    };

    const ret = try expr.eval(arena_instance.allocator());
    const exp = Ast.Node{ .variable = "x y" };
    try std.testing.expect(ret.eql(exp));
}

test "eval function" {
    var debug = std.heap.DebugAllocator(.{}).init;
    defer _ = debug.deinit();
    var arena_instance = std.heap.ArenaAllocator.init(debug.allocator());
    defer arena_instance.deinit();

    const expr = Ast.Node{
        .function = .{
            .arg = "x",
            .body = @constCast(&Ast.Node{
                .variable = "x",
            }),
        },
    };

    const ret = try expr.eval(arena_instance.allocator());
    const exp = Ast.Node{
        .function = .{
            .arg = "x",
            .body = @constCast(&Ast.Node{
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

    const expr = Ast.Node{
        .app = .{
            .lhs = @constCast(&Ast.Node{
                .function = .{
                    .arg = "x",
                    .body = @constCast(&Ast.Node{
                        .variable = "x",
                    }),
                },
            }),
            .rhs = @constCast(&Ast.Node{ .variable = "y" }),
        },
    };

    const ret = try expr.eval(arena_instance.allocator());
    const exp = Ast.Node{ .variable = "y" };
    try std.testing.expect(ret.eql(exp));
}
