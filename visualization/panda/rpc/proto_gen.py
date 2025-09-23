# 这段代码使用 grpc_tools 中的 protoc 模块来编译 Protocol Buffers 文件 (.proto 文件)
# 生成相应的 Python 文件,这些文件用于在 Python 中实现 gRPC 服务和客户端

from grpc_tools import protoc

protoc.main(
    (
        '',
        '-I.',
        '--python_out=.',
        '--grpc_python_out=.',
        './rviz.proto',
    )
)
