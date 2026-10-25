// IPinyin for macOS —— InputMethodKit 输入法主程序。
// 引擎本体在 tools/macos_server.py（Python，随包内置于 Resources/engine），
// 这里只做三件事：注册输入源、拉起引擎桥、声明 IMKServer 连接。
import Cocoa
import Carbon
import InputMethodKit

let kConnectionName = "IPinyinServerConnection"

final class AppDelegate: NSObject, NSApplicationDelegate {
    var server: IMKServer?

    func applicationDidFinishLaunching(_ notification: Notification) {
        EngineProcess.ensureRunning()
        server = IMKServer(name: kConnectionName, bundleIdentifier: Bundle.main.bundleIdentifier)
        NSLog("ipinyin: IMKServer 就绪（%@）", kConnectionName)
    }

    func applicationWillTerminate(_ notification: Notification) {
        EngineProcess.stop()
    }
}

// --register / --select：注册并启用输入源（构建脚本与用户补救都可用）
enum RegisterHelper {
    static func run(select: Bool) {
        let url = URL(fileURLWithPath: Bundle.main.bundlePath) as CFURL
        let status = TISRegisterInputSource(url)
        print("TISRegisterInputSource: \(status)")
        guard status == noErr || status == 0 else { return }
        if select {
            enableAndSelect()
        }
    }

    static func enableAndSelect() {
        guard let bundleID = Bundle.main.bundleIdentifier else { return }
        let filter = [kTISPropertyBundleID as String: bundleID] as CFDictionary
        guard let list = TISCreateInputSourceList(filter, false)?.takeRetainedValue()
                as? [TISInputSource], !list.isEmpty else {
            print("输入源尚未出现（可能需要注销重登一次）")
            return
        }
        for source in list {
            TISEnableInputSource(source)
            let sel = TISSelectInputSource(source)
            if let name = TISGetInputSourceProperty(source, kTISPropertyLocalizedName) {
                print("已启用并选中：\(Unmanaged<CFString>.fromOpaque(name).takeUnretainedValue())（status=\(sel)）")
            }
        }
    }
}

if CommandLine.arguments.contains("--register") {
    RegisterHelper.run(select: CommandLine.arguments.contains("--select"))
    exit(0)
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.accessory)
app.run()
