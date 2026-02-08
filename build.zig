const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.resolveTargetQuery(.{});
    const optimize = b.standardOptimizeOption(.{});

    const lambda_mod = b.createModule(.{
        .root_source_file = b.path("lambda.zig"),
        .target = target,
        .optimize = optimize,
    });

    const exe = b.addExecutable(.{
        .name = "lambda",
        .root_module = lambda_mod,
    });

    b.installArtifact(exe);

    const tests = b.addTest(.{ .root_module = lambda_mod });
    const run_tests = b.addRunArtifact(tests);
    b.step("test", "Run tests").dependOn(&run_tests.step);
}
