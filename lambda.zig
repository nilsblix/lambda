const std = @import("std");
const Allocator = std.mem.Allocator;
const Writer = std.Io.Writer;

const safety_bound: usize = 100_000;

const Ast = struct {
    root_ref: Node.Ref = 0,
    gc: GC = .{},

    fn deinit(self: *Ast, gpa: Allocator) void {
        self.gc.deinit(gpa);
    }

    fn clear(self: *Ast, gpa: Allocator) void {
        self.deinit(gpa);
        self.* = Ast{};
    }

    fn root(self: *const Ast) ?Node {
        const r = self.gc.get(self.root_ref) orelse return null;
        return r.*;
    }

    pub fn format(self: Ast, w: *Writer) Writer.Error!void {
        if (self.root()) |r| {
            try r.format(&self.gc, w);
        } else {
            try w.print("no root", .{});
        }
    }

    const GC = struct {
        buf: std.ArrayList(Node) = .empty,

        fn get(self: *const GC, ref: Node.Ref) ?*Node {
            if (ref >= self.buf.items.len) return null;
            return &self.buf.items[ref];
        }

        fn make(self: *GC, gpa: Allocator, node: Node) Allocator.Error!Node.Ref {
            try self.buf.append(gpa, node);
            return self.buf.items.len - 1;
        }

        fn deinit(self: *GC, gpa: Allocator) void {
            self.buf.deinit(gpa);
        }
    };

    const Node = union(enum) {
        const Ref = usize;

        const Function = struct {
            arg: []const u8,
            body: Ref,
        };

        const App = struct {
            lhs: Ref,
            rhs: Ref,
        };

        variable: []const u8,
        function: Function,
        app: App,

        fn printable(self: Node, gpa: Allocator, gc: *GC) Allocator.Error!Ast {
            const tmp = try gc.make(gpa, self);
            return Ast{
                .root_ref = tmp,
                .gc = gc.*,
            };
        }

        fn format(self: Node, gc: *const GC, w: *Writer) Writer.Error!void {
            switch (self) {
                .variable => |v| try w.print("{s}", .{v}),
                .function => |f| {
                    try w.print("\\{s}.", .{f.arg});

                    const body = gc.get(f.body) orelse {
                        try w.print("['{d}' not found]", .{f.body});
                        return;
                    };

                    try body.format(gc, w);
                },
                .app => |app| {
                    const lhs = gc.get(app.lhs) orelse {
                        try w.print("['{d}' not found]", .{app.lhs});
                        return;
                    };
                    try w.print("(", .{});
                    try lhs.format(gc, w);
                    try w.print(" ", .{});
                    const rhs = gc.get(app.rhs) orelse {
                        try w.print("['{d}' not found]", .{app.rhs});
                        return;
                    };
                    try rhs.format(gc, w);
                    try w.print(")", .{});
                },
            }
        }

        /// `body` needs to be freed after calling this, as all heap allocated
        /// members get duped in this function.
        ///
        /// All members of the returned Node are heap allocated, while the top
        /// level Node is copied, thus on the stack.
        ///
        /// (\param.body) arg
        fn replace(
            gpa: Allocator,
            gc: *GC,
            param: []const u8,
            body_ref: Ref,
            arg_ref: Ref,
        ) Allocator.Error!Ref {
            const body = body: {
                const ret = gc.get(body_ref) orelse unreachable;
                break :body ret.*;
            };
            switch (body) {
                // (\param.v) arg
                .variable => {
                    if (body == .variable and std.mem.eql(u8, body.variable, param)) return arg_ref;
                    return body_ref;
                },
                // (\param.\inner_arg.inner_body) arg
                // We need to replace all occurences of `param` in `inner_arg`
                // and `inner_body` with `arg`.
                .function => |inner| {
                    const new_inner_arg = new_inner_arg: {
                        // TODO: Do we really have to heap allocate this
                        // temporary thing? Feels like this is just a
                        // limitation of the replace signature being dependant
                        // on a bunch of refs.
                        const tmp = try gc.make(gpa, .{ .variable = inner.arg });
                        const ref = try replace(gpa, gc, param, tmp, arg_ref);
                        const node = gc.get(ref);
                        break :new_inner_arg node.?.variable;
                    };

                    const new_inner_body = try replace(gpa, gc, param, inner.body, arg_ref);
                    const ret = Node{
                        .function = .{ .arg = new_inner_arg, .body = new_inner_body },
                    };
                    return try gc.make(gpa, ret);
                },
                // (\param.(lhs rhs)) arg
                // We need to replace all occurences of `param` in `lhs` and `rhs`
                // with `arg`.
                .app => |app| {
                    const new_lhs = try replace(gpa, gc, param, app.lhs, arg_ref);
                    const new_rhs = try replace(gpa, gc, param, app.rhs, arg_ref);
                    const ret = Node{
                        .app = .{ .lhs = new_lhs, .rhs = new_rhs },
                    };
                    return try gc.make(gpa, ret);
                },
            }
        }

        const EvalError = error{ExpectedFunction} || Allocator.Error;

        fn eval(self: *const Node, gpa: Allocator, gc: *GC) EvalError!Ref {
            switch (self.*) {
                // Snapshot before append: gc.make can grow and relocate storage.
                .variable, .function => {
                    const node = self.*;
                    return try gc.make(gpa, node);
                },
                .app => |app| {
                    // We know that all nodes get created via gc.make,
                    // therefore invalid state doesn't exist.
                    const lhs = (gc.get(app.lhs) orelse unreachable).*;
                    const function_ref = try lhs.eval(gpa, gc);

                    const rhs = (gc.get(app.rhs) orelse unreachable).*;
                    const replacement_ref = try rhs.eval(gpa, gc);

                    const f = gc.get(function_ref) orelse unreachable;
                    if (f.* != .function) return error.ExpectedFunction;
                    return try replace(gpa, gc, f.function.arg, f.function.body, replacement_ref);
                },
            }
        }
    };

    fn eval(self: *Ast, gpa: Allocator) Node.EvalError!?Node {
        const r = self.gc.get(self.root_ref) orelse return null;
        var ref = try r.eval(gpa, &self.gc);

        for (0..safety_bound) |_| {
            const ptr = self.gc.get(ref) orelse return null;
            if (ptr.* != .app) return ptr.*;
            const next = ptr.*;
            ref = try next.eval(gpa, &self.gc);
        } else @panic("loop safety counter exceeded");
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
    comment,
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
            ';' => .comment,
            0 => .end,
            else => .{ .ident = "" },
        };
    }

    fn kindEql(a: Token, b: Token) bool {
        return std.mem.eql(u8, @tagName(a), @tagName(b));
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

    fn peekByte(self: *Lexer) ?u8 {
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
            .eq, .lambda, .oparen, .cparen, .separator, .comment, .end, .illegal => return tok,
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
                        b = self.peekByte() orelse break;
                        if (Token.match(b) != .ident) break;
                        _ = self.byteAndAdvance();
                    }
                    return .{ .ident = self.content[start..self.cur] };
                }
            },
        }
    }

    fn peek(self: *Lexer) Token {
        const cur = self.cur;
        const tok = self.next();
        self.cur = cur;
        return tok;
    }

    // Does not check the inside of `ident`. Simply expecting the kind.
    fn expect(self: *Lexer, expected: Token) ?void {
        if (!self.peek().kindEql(expected)) return null;
    }

    fn consume(self: *Lexer) void {
        _ = self.next();
    }
};

fn parseFunction(gpa: Allocator, gc: *Ast.GC, l: *Lexer) Allocator.Error!?Ast.Node.Ref {
    l.expect(.{ .ident = "" }) orelse {
        std.log.err("unexpected token. expected identifier, found: '{s}'\n", .{@tagName(l.next())});
        return null;
    };
    const arg = l.next().ident;

    l.expect(.separator) orelse {
        std.log.err("unexpected token. expected separator, found: '{s}'\n", .{@tagName(l.next())});
        return null;
    };
    l.consume();

    const state = l.*;
    var a = l.next();
    var b = l.peek();
    l.* = state;

    var body: Ast.Node.Ref = undefined;
    if (a.kindEql(.{ .ident = "" }) and b.kindEql(.separator)) {
        body = try parseFunction(gpa, gc, l) orelse return null;
    } else {
        body = try parseExpression(gpa, gc, l) orelse return null;
    }

    return try gc.make(gpa, .{ .function = .{ .arg = arg, .body = body } });
}

fn parseIdentifier(gpa: Allocator, gc: *Ast.GC, l: *Lexer, i: []const u8) Allocator.Error!?Ast.Node.Ref {
    // We have a couple of cases.
    // 1. i...) or ; or eof  ; return variable
    // 2. i variable ; return application of i and var
    // 3. i (lambda) ; return application of i and function
    for (0..safety_bound) |_| {
        if (l.peek() == .space) {
            l.consume();
            continue;
        }
        break;
    } else @panic("loop safety counter exceeded");

    var ret = try gc.make(gpa, .{ .variable = i });

    // We have to iterate as to encapsulate with left-priority all arguments to
    // a function application.
    // ex: f x y z == ((f x) y) z
    for (0..safety_bound) |_| {
        const peeked = l.peek();

        // case 1
        if (peeked == .cparen or peeked == .comment or peeked == .end) {
            return ret;
        }

        // case 2.
        if (peeked.kindEql(.{ .ident = "" })) {
            const variable = l.next().ident;
            const variable_ref = try gc.make(gpa, .{ .variable = variable });
            ret = try gc.make(gpa, .{ .app = .{ .lhs = ret, .rhs = variable_ref } });
            continue;
        }

        // case 3.
        if (peeked.kindEql(.lambda) or peeked.kindEql(.oparen)) {
            l.consume(); // lambda symbol
            const func = try parseFunction(gpa, gc, l) orelse return null;
            ret = try gc.make(gpa, .{ .app = .{ .lhs = ret, .rhs = func } });
            continue;
        }

        break;
    } else @panic("loop safety counter exceeded");

    return ret;
}

fn parseExpression(gpa: Allocator, gc: *Ast.GC, l: *Lexer) Allocator.Error!?Ast.Node.Ref {
    const next = l.next();
    switch (next) {
        .ident => |i| return try parseIdentifier(gpa, gc, l, i),
        .lambda => return parseFunction(gpa, gc, l),
        .oparen => {
            // Inner gets allocated via gc elsewhere, therefore no need to call
            // `make`.
            const inner = try parseExpression(gpa, gc, l) orelse return null;

            for (0..safety_bound) |_| {
                if (l.peek() == .space) {
                    l.consume();
                    continue;
                }
                break;
            } else @panic("loop safety counter exceeded");

            // Either we have something else inside this paren, or we don't
            // 1. (inner)     ; unnessecary wrapping of variable/application, return inner
            // 2. (inner rhs) ; return application
            // 3. (inner) rhs ; inner is a function => return application rhs

            if (l.peek() == .cparen) {
                if (gc.get(inner).?.* == .function) {
                    // case 3.
                    l.consume(); // cparen
                    // If there is no rhs, then we are equivalent to case 1,
                    // therefore return inner.
                    const rhs = try parseExpression(gpa, gc, l) orelse return inner;
                    return try gc.make(gpa, .{ .app = .{ .lhs = inner, .rhs = rhs } });
                }

                // case 1.
                l.consume(); // cparen
                return inner;
            }

            // case 2.
            const rhs = try parseExpression(gpa, gc, l) orelse return null;
            return try gc.make(gpa, .{ .app = .{ .lhs = inner, .rhs = rhs } });
        },
        .eq, .cparen, .separator, .space, .comment, .end, .illegal => {
            std.log.err("unexpected token. expected a primary token, found: '{s}'\n", .{@tagName(next)});
            return null;
        },
    }
}

const Parsed = struct { Ast, usize };

fn parse(gpa: Allocator, input: []const u8) Allocator.Error!?Parsed {
    var gc = Ast.GC{};
    var l = Lexer.init(input);
    const root = try parseExpression(gpa, &gc, &l) orelse return null;
    return .{ Ast{ .root_ref = root, .gc = gc }, l.cur };
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}).init;
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const f = std.fs.File.stdin();
    defer f.close();

    var reader_buf: [4096]u8 = undefined;
    var reader = f.reader(&reader_buf);

    while (true) {
        std.debug.print("@> ", .{});
        const input = try reader.interface.takeDelimiter('\n') orelse continue;
        if (std.mem.eql(u8, input, ":quit")) break;

        var ast, _ = try parse(alloc, input) orelse continue;
        defer ast.deinit(alloc);

        const res = try ast.eval(alloc) orelse {
            std.debug.print("info: did not evaluate to a proper result\n", .{});
            continue;
        };

        std.debug.print("{f}\n", .{try res.printable(alloc, &ast.gc)});
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

test "parse" {
    var debug = std.heap.DebugAllocator(.{}).init;
    defer _ = debug.deinit();
    const gpa = debug.allocator();

    const expect = struct {
        fn expect(alloc: Allocator, input: []const u8, expected: Ast) !void {
            var writer_buf: [8192]u8 = undefined;
            var w = std.Io.Writer.fixed(&writer_buf);

            var ast, _ = try parse(alloc, input) orelse return error.Unexpected;
            defer ast.deinit(alloc);

            try w.print("{f}", .{ast});
            const got_end = w.end;
            const got = writer_buf[0..got_end];

            try w.print("{f}", .{expected});
            const exp = writer_buf[got_end..w.end];
            try std.testing.expectEqualStrings(exp, got);
        }
    }.expect;

    var ast = Ast{};
    errdefer ast.deinit(gpa);

    ast.root_ref = try ast.gc.make(gpa, .{ .variable = "var" });
    try expect(gpa, "   var     ", ast);
    ast.clear(gpa);

    var input: []const u8 =
        \\\x.function_body
    ;
    _ = try ast.gc.make(gpa, .{ .variable = "function_body" });
    ast.root_ref = try ast.gc.make(gpa, .{ .function = .{ .arg = "x", .body = 0 } });
    try expect(gpa, input, ast);
    ast.clear(gpa);

    input =
        \\ (\function_arg.function_body) apped
    ;
    _ = try ast.gc.make(gpa, .{ .variable = "function_body" });
    _ = try ast.gc.make(gpa, .{ .function = .{ .arg = "function_arg", .body = 0 } });
    _ = try ast.gc.make(gpa, .{ .variable = "apped" });
    ast.root_ref = try ast.gc.make(gpa, .{ .app = .{ .lhs = 1, .rhs = 2 } });
    try expect(gpa, input, ast);
    ast.clear(gpa);

    input =
        \\ function app_this
    ;
    _ = try ast.gc.make(gpa, .{ .variable = "function" });
    _ = try ast.gc.make(gpa, .{ .variable = "app_this" });
    ast.root_ref = try ast.gc.make(gpa, .{ .app = .{ .lhs = 0, .rhs = 1 } });
    try expect(gpa, input, ast);
    ast.clear(gpa);

    input =
        \\ function ; app_this
    ;
    ast.root_ref = try ast.gc.make(gpa, .{ .variable = "function" });
    try expect(gpa, input, ast);
    ast.clear(gpa);

    input =
        \\ (function app_this)
    ;
    _ = try ast.gc.make(gpa, .{ .variable = "function" });
    _ = try ast.gc.make(gpa, .{ .variable = "app_this" });
    ast.root_ref = try ast.gc.make(gpa, .{ .app = .{ .lhs = 0, .rhs = 1 } });
    try expect(gpa, input, ast);
    ast.clear(gpa);

    input =
        \\ ((((  var)  )))
    ;
    ast.root_ref = try ast.gc.make(gpa, .{ .variable = "var" });
    try expect(gpa, input, ast);
    ast.clear(gpa);

    input =
        \\ \x.\y.\f.f x y
    ;
    {
        const x = try ast.gc.make(gpa, .{ .variable = "x" });
        const y = try ast.gc.make(gpa, .{ .variable = "y" });
        const f = try ast.gc.make(gpa, .{ .variable = "f" });
        const fx = try ast.gc.make(gpa, .{ .app = .{ .lhs = f, .rhs = x } });
        const fxy = try ast.gc.make(gpa, .{ .app = .{ .lhs = fx, .rhs = y } });
        const ffxy = try ast.gc.make(gpa, .{ .function = .{ .arg = "f", .body = fxy } });
        const yffxy = try ast.gc.make(gpa, .{ .function = .{ .arg = "y", .body = ffxy } });
        ast.root_ref = try ast.gc.make(gpa, .{ .function = .{ .arg = "x", .body = yffxy } });
        try expect(gpa, input, ast);
        ast.clear(gpa);
    }

    input =
        \\ \x.some_return \wow.some.more
    ;
    _ = try ast.gc.make(gpa, .{ .variable = "some_return" });
    _ = try ast.gc.make(gpa, .{ .variable = "more" });
    _ = try ast.gc.make(gpa, .{ .function = .{ .arg = "some", .body = 1 } });
    _ = try ast.gc.make(gpa, .{ .function = .{ .arg = "wow", .body = 2 } });
    _ = try ast.gc.make(gpa, .{ .app = .{ .lhs = 0, .rhs = 3 } });
    _ = try ast.gc.make(gpa, .{ .function = .{ .arg = "x", .body = 4 } });
    ast.root_ref = 5;
    try expect(gpa, input, ast);
    ast.root_ref = 3;
    try expect(gpa, input[15..], ast);
    ast.clear(gpa);

    input =
        \\ (\x.(x x)) (\x.(x x))
    ;
    const x = try ast.gc.make(gpa, .{ .variable = "x" });
    const x_x = try ast.gc.make(gpa, .{ .app = .{ .lhs = x, .rhs = x } });
    const rec = try ast.gc.make(gpa, .{ .function = .{ .arg = "x", .body = x_x } });
    ast.root_ref = try ast.gc.make(gpa, .{ .app = .{ .lhs = rec, .rhs = rec } });
    try expect(gpa, input, ast);
    ast.clear(gpa);
}

test "eval" {
    var debug = std.heap.DebugAllocator(.{}).init;
    defer _ = debug.deinit();
    const gpa = debug.allocator();

    const expect = struct {
        fn expect(alloc: Allocator, input: []const u8, expected: Ast) !void {
            var writer_buf: [8192]u8 = undefined;
            var w = std.Io.Writer.fixed(&writer_buf);

            var ast, _ = try parse(alloc, input) orelse return error.Unexpected;
            defer ast.deinit(alloc);

            var res = try ast.eval(alloc) orelse return error.Unexpected;
            const printable = try res.printable(alloc, &ast.gc);

            try w.print("{f}", .{printable});
            const got_end = w.end;
            const got = writer_buf[0..got_end];

            try w.print("{f}", .{expected});
            const exp = writer_buf[got_end..w.end];
            try std.testing.expectEqualStrings(exp, got);
        }
    }.expect;

    var ast = Ast{};
    errdefer ast.deinit(gpa);

    var input: []const u8 =
        \\ variable
    ;
    ast.root_ref = try ast.gc.make(gpa, .{ .variable = "variable" });
    try expect(gpa, input, ast);
    ast.clear(gpa);

    input =
        \\ \x.x          
    ;
    _ = try ast.gc.make(gpa, .{ .variable = "x" });
    ast.root_ref = try ast.gc.make(gpa, .{ .function = .{ .arg = "x", .body = 0 } });
    try expect(gpa, input, ast);
    ast.clear(gpa);

    input =
        \\ (\x.x) y
    ;
    ast.root_ref = try ast.gc.make(gpa, .{ .variable = "y" });
    try expect(gpa, input, ast);
    ast.clear(gpa);

    input =
        \\ (\x.f.x) y
    ;
    _ = try ast.gc.make(gpa, .{ .variable = "y" });
    ast.root_ref = try ast.gc.make(gpa, .{ .function = .{ .arg = "f", .body = 0 } });
    try expect(gpa, input, ast);
    ast.clear(gpa);

    input =
        \\ \x.y.f.f x y
    ;
    const x = try ast.gc.make(gpa, .{ .variable = "x" });
    const y = try ast.gc.make(gpa, .{ .variable = "y" });
    const f = try ast.gc.make(gpa, .{ .variable = "f" });
    const fx = try ast.gc.make(gpa, .{ .app = .{ .lhs = f, .rhs = x } });
    const fxy = try ast.gc.make(gpa, .{ .app = .{ .lhs = fx, .rhs = y } });
    const ffxy = try ast.gc.make(gpa, .{ .function = .{ .arg = "f", .body = fxy } });
    const yffxy = try ast.gc.make(gpa, .{ .function = .{ .arg = "y", .body = ffxy } });
    ast.root_ref = try ast.gc.make(gpa, .{ .function = .{ .arg = "x", .body = yffxy } });
    try expect(gpa, input, ast);
    ast.clear(gpa);
}
