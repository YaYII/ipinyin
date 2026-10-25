// 引擎桥接客户端（unix socket 行协议）与引擎进程管理。
import Foundation

struct EngineReply {
    let handled: Bool
    let commit: String?
    let preedit: String
    let visible: Bool
    let candidates: [String]
    let words: [String]
    let highlight: Int
    let page: Int
    let pageSize: Int

    init(_ dict: [String: Any]) {
        handled = dict["handled"] as? Bool ?? false
        commit = dict["commit"] as? String
        preedit = dict["preedit"] as? String ?? ""
        visible = dict["visible"] as? Bool ?? false
        candidates = dict["candidates"] as? [String] ?? []
        words = dict["words"] as? [String] ?? []
        highlight = dict["highlight"] as? Int ?? 0
        page = dict["page"] as? Int ?? 0
        pageSize = dict["page_size"] as? Int ?? 9
    }
}

final class EngineClient {
    static let shared = EngineClient()
    private var fd: Int32 = -1
    private var buffer = Data()

    var socketPath: String {
        let appSupport = NSSearchPathForDirectoriesInDomains(.applicationSupportDirectory, .userDomainMask, true).first!
        return appSupport + "/ipinyin/engine.sock"
    }

    func event(keyval: Int, state: Int) -> EngineReply? {
        return request(["t": "event", "keyval": keyval, "state": state])
    }

    func select(index: Int) -> EngineReply? {
        return request(["t": "select", "index": index])
    }

    private func request(_ obj: [String: Any]) -> EngineReply? {
        for _ in 0..<2 {
            if !connected() {
                EngineProcess.ensureRunning()
                usleep(250_000)
                if !connected() { continue }
            }
            if let dict = roundTrip(obj) {
                return EngineReply(dict)
            }
            disconnect()               // 引擎死了/应答断了：重连重来一次
        }
        return nil
    }

    private func connected() -> Bool {
        if fd >= 0 { return true }
        let sock = socket(AF_UNIX, SOCK_STREAM, 0)
        guard sock >= 0 else { return false }
        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let path = Array(socketPath.utf8)
        guard path.count + 1 < MemoryLayout.size(ofValue: addr.sun_path) else {
            close(sock)
            return false
        }
        withUnsafeMutablePointer(to: &addr.sun_path) { dest in
            path.withUnsafeBufferPointer { src in
                memcpy(dest, src.baseAddress!, path.count)
            }
        }
        let ok = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                connect(sock, $0, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        guard ok == 0 else {
            close(sock)
            return false
        }
        var tv = timeval(tv_sec: 1, tv_usec: 0)
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))
        fd = sock
        buffer.removeAll()
        return true
    }

    private func roundTrip(_ obj: [String: Any]) -> [String: Any]? {
        guard JSONSerialization.isValidJSONObject(obj),
              var payload = try? JSONSerialization.data(withJSONObject: obj) else { return nil }
        payload.append(0x0A)
        let sent = payload.withUnsafeBytes { raw in
            Darwin.send(fd, raw.baseAddress, raw.count, Int32(0))
        }
        guard sent == payload.count else { return nil }

        var deadline = Date().addingTimeInterval(1.5)
        while Date() < deadline {
            if let nl = buffer.firstIndex(of: 0x0A) {
                let line = buffer.subdata(in: buffer.startIndex..<nl)
                buffer.removeSubrange(buffer.startIndex...nl)
                if let dict = try? JSONSerialization.jsonObject(with: line) as? [String: Any] {
                    return dict
                }
                return nil
            }
            var chunk = [UInt8](repeating: 0, count: 65536)
            let n = read(fd, &chunk, chunk.count)
            if n <= 0 { return nil }
            buffer.append(contentsOf: chunk[0..<n])
        }
        return nil
    }

    private func disconnect() {
        if fd >= 0 { close(fd) }
        fd = -1
        buffer.removeAll()
    }
}

enum EngineProcess {
    private static var process: Process?

    static func ensureRunning() {
        if let p = process, p.isRunning { return }
        guard let engineDir = Bundle.main.resourceURL?.appendingPathComponent("engine") else { return }
        let appSupport = NSSearchPathForDirectoriesInDomains(.applicationSupportDirectory, .userDomainMask, true).first! + "/ipinyin"
        try? FileManager.default.createDirectory(atPath: appSupport, withIntermediateDirectories: true)

        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        p.arguments = [engineDir.appendingPathComponent("tools/macos_server.py").path,
                       "--data", engineDir.appendingPathComponent("data").path]
        if let log = FileHandle(forWritingAtPath: appSupport + "/engine.log") {
            p.standardError = log
            p.standardOutput = log
        }
        do {
            try p.run()
            process = p
        } catch {
            NSLog("ipinyin: 无法启动引擎桥：\(error)")
        }
    }

    static func stop() {
        process?.terminate()
        process = nil
    }
}
