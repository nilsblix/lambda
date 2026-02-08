// Single file REPL for a Lambda Calculus interpreter.
//
// ### Syntax:
//
// A *variable* is some sequence of characters. For example: `\x.y` denotes a
// function where the argument is a variable `x` and returns a variable `y`.
// Variables can be any length, and can overload semantic tokens by wrapping
// the variable name in @"...". Examples:
//
// `x`, `var`, `anything`, `@"some variable"`, `@"\wow.\is this a variable?"`.
//
// A *function* takes the form: `\x.y` where the `x` variable is the function's
// argument, and `y` is the returned expression. Note that `y` doesn't have to
// be a variable, which enables important functional concepts, such as
// currying. Examples:
//
// `\x.x`         - x -> x (the id function)
// `\x.y`         - x -> y
// `\x.y.z`       - x, y -> z
// `\x.y.f.f x y` - When applied to two arguments, it creates a pair.
//
// An *application* of a function takes the form: `(f x)` where `f` is the
// function to apply the argument `x` to. Applications are usually
// syntactically similar to S-expressions, with the notable exceptions of being
// inside function returns, such as in the Pair example above, where `f x y` in
// the function return is an application of `f` with the arguments `x` and `y`.
// Examples:
//
// `(f x)`   - f(x)
// `(g x y)` - g(x, y). Note that `g` has to be defined as a curried function,
// i.e `g = \x.y.z`, thus making `(g x)` equal to `\y.z`.
//
//
// ### Examples
//
// With this system, we can define simple types, such as booleans:
//
// `true = \x.y.x`
// `false = \x.y.y`.
//
// But, why does these two functions behave as booleans? Lets try to use them
// in some if/else statement. But first lets introduce an `if` function.
//
// `if = \x.x` (i.e the id function)
//
// Lets see what happens when we evaluate the following expression:
//
// `if true then else`
// == `(\x.x \x.y.x) then else`
// == `(\x.y.x then) else`
// == `(\y.then else)`
// == `then`
//
// Wow, our entire expression got evaluated down to `then`. You can yourself
// try it with `false`, and see that it gets cooked down to `else`.
//
// Let us now take the `pair` example from above, and try to evaluate it with
// two arguments:
//
// `pair = \x.y.f.f x y`
// `pair 12 13` == `\f.f 12 13`.
//
// We can define `fst` and `snd` functions to extract these two items in our pair:
//
// `fst = \p.p true`
// `snd = \p.p false`
//
// Lets try them on our pair:
//
// `(fst (pair 12 13))`
// == `((\p.p true) (\f.f 12 13))`
// == `(\f.f 12 13) true`
// == `true 12 13`
// == `12`
//
// `(snd (pair 12 13))`
// == `((\p.p false) (\f.f 12 13))`
// == `(\f.f 12 13) false`
// == `false 12 13`
// == `13`

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

fn nextExpression(gpa: Allocator, gc: *Ast.GC, l: *Lexer) Allocator.Error!?Ast.Node.Ref {
    switch (l.next()) {
        .ident => |i| return try gc.make(gpa, .{ .variable = i }),
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

            const body = try nextExpression(gpa, gc, l) orelse return null;
            return try gc.make(gpa, Ast.Node{ .function = .{ .arg = arg, .body = body } });
        },
        .oparen => {
            const f = try nextExpression(gpa, gc, l) orelse return null;

            const arg = try nextExpression(gpa, gc, l) orelse return null;

            const end = l.byteAndAdvance();
            if (end == null or Token.match(end.?) != .cparen) {
                const c = end orelse 0;
                std.log.err("expected ')' to close application, found `{c}`", .{c});
                return null;
            }

            return try gc.make(gpa, Ast.Node{ .app = .{ .lhs = f, .rhs = arg } });
        },
        .eq, .cparen, .separator, .space, .end, .illegal => return null,
    }
}

const Parsed = struct { Ast, usize };

fn parse(gpa: Allocator, input: []const u8) Allocator.Error!?Parsed {
    var gc = Ast.GC{};
    var l = Lexer.init(input);
    const root = try nextExpression(gpa, &gc, &l) orelse return null;
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

test "parse expression and deinit" {
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
        \\ \x.function_body
    ;
    _ = try ast.gc.make(gpa, .{ .variable = "function_body" });
    ast.root_ref = try ast.gc.make(gpa, .{ .function = .{ .arg = "x", .body = 0 } });
    try expect(gpa, input, ast);
    ast.clear(gpa);

    input =
        \\ (\function_arg.function_body apped)
    ;
    _ = try ast.gc.make(gpa, .{ .variable = "function_body" });
    _ = try ast.gc.make(gpa, .{ .function = .{ .arg = "function_arg", .body = 0 } });
    _ = try ast.gc.make(gpa, .{ .variable = "apped" });
    ast.root_ref = try ast.gc.make(gpa, .{ .app = .{ .lhs = 1, .rhs = 2 } });
    try expect(gpa, input, ast);
    ast.clear(gpa);

    input =
        \\ \x.some_return \wow.\some.more
    ;
    _ = try ast.gc.make(gpa, .{ .variable = "some_return" });
    _ = try ast.gc.make(gpa, .{ .function = .{ .arg = "x", .body = 0 }});
    _ = try ast.gc.make(gpa, .{ .variable = "more" });
    _ = try ast.gc.make(gpa, .{ .function = .{ .arg = "some", .body = 2 }});
    _ = try ast.gc.make(gpa, .{ .function = .{ .arg = "wow", .body = 3 }});
    ast.root_ref = 1;
    try expect(gpa, input, ast);
    ast.root_ref = 4;
    try expect(gpa, input[15..], ast);
    ast.clear(gpa);

    input =
        \\ (\x.(x x) \x.(x x))
    ;
    const x = try ast.gc.make(gpa, .{ .variable = "x" });
    const x_x = try ast.gc.make(gpa, .{ .app = .{ .lhs = x, .rhs = x }});
    const rec = try ast.gc.make(gpa, .{ .function = .{ .arg = "x", .body = x_x }});
    ast.root_ref = try ast.gc.make(gpa, .{ .app = .{ .lhs = rec, .rhs = rec }});
    try expect(gpa, input, ast);
    ast.clear(gpa);
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
    ast.root_ref = try ast.gc.make(gpa, .{ .function = .{ .arg = "x", .body = 0 }});
    try expect(gpa, input, ast);
    ast.clear(gpa);

    input =
        \\ (\x.x y)
    ;
    ast.root_ref = try ast.gc.make(gpa, .{ .variable = "y" });
    try expect(gpa, input, ast);
    ast.clear(gpa);

    input =
        \\ (\x.\f.x y)
    ;
    _ = try ast.gc.make(gpa, .{ .variable = "y" });
    ast.root_ref = try ast.gc.make(gpa, .{ .function = .{ .arg = "f", .body = 0 }});
    try expect(gpa, input, ast);
    ast.clear(gpa);
}
