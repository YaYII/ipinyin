// IPinyin 打字条：不公证也能在任意 App 里用 IPinyin 打字。
//
// Apple 只拦「系统输入源」注册，不拦本地 app 往前台应用注入文字
// （Espanso / CapslockWriter 同款思路）：⌃⌥空格 唤出悬浮输入条 →
// 打拼音选词 → 上屏文字用 CGEvent 打进之前那个 App 的光标处。
// 需要「辅助功能」权限（首次启动系统会弹授权引导）。
import Cocoa
import Carbon

final class BarPanel: NSPanel {
    override var canBecomeKey: Bool { true }
}

// ---- 键值映射（与 Controller.swift 同源） ---------------------------------

private let SHIFT_MASK = 1 << 0

private func mapKeyval(_ event: NSEvent) -> Int? {
    switch Int(event.keyCode) {
    case 51: return 0xFF08      // backspace
    case 36: return 0xFF0D      // return
    case 76: return 0xFF8D      // keypad enter
    case 53: return 0xFF1B      // esc
    case 48: return 0xFF09      // tab
    case 116: return 0xFF55     // page up
    case 121: return 0xFF56     // page down
    case 126: return 0xFF52     // up
    case 125: return 0xFF54     // down
    case 123: return 0xFF51     // left
    case 124: return 0xFF53     // right
    default: break
    }
    guard let chars = event.charactersIgnoringModifiers, let first = chars.first,
          let scalar = first.asciiValue else { return nil }
    return Int(scalar)
}

// ---- 把文字打进前台应用 ---------------------------------------------------

enum TypeInserter {
    static func type(_ text: String, delayMs: UInt32 = 8) {
        for ch in text.unicodeScalars {
            var units = Array(String(ch).utf16)
            let down = CGEvent(keyboardEventSource: nil, virtualKey: 0, keyDown: true)!
            down.keyboardSetUnicodeString(stringLength: units.count, unicodeString: &units)
            down.post(tap: .cghidEventTap)
            usleep(delayMs * 1000)
            let up = CGEvent(keyboardEventSource: nil, virtualKey: 0, keyDown: false)!
            up.keyboardSetUnicodeString(stringLength: units.count, unicodeString: &units)
            up.post(tap: .cghidEventTap)
            usleep(2000)
        }
    }

    static func pressKey(_ keyCode: CGKeyCode) {
        let down = CGEvent(keyboardEventSource: nil, virtualKey: keyCode, keyDown: true)!
        down.post(tap: .cghidEventTap)
        usleep(8000)
        let up = CGEvent(keyboardEventSource: nil, virtualKey: keyCode, keyDown: false)!
        up.post(tap: .cghidEventTap)
    }
}

// ---- 输入条视图 ------------------------------------------------------------

final class BarView: NSView {
    private let engine = EngineClient.shared
    private var label: NSTextField!
    private weak var panel: BarPanel?
    private var previousApp: NSRunningApplication?

    init(panel: BarPanel) {
        self.panel = panel
        super.init(frame: NSRect(x: 0, y: 0, width: 620, height: 34))
        label = NSTextField(labelWithString: "")
        label.frame = NSRect(x: 10, y: 7, width: 600, height: 20)
        label.font = NSFont.monospacedSystemFont(ofSize: 14, weight: .regular)
        label.textColor = .textColor
        addSubview(label)
        wantsLayer = true
        layer?.backgroundColor = NSColor.textBackgroundColor.withAlphaComponent(0.96).cgColor
        layer?.cornerRadius = 8
        layer?.borderWidth = 1
        layer?.borderColor = NSColor.separatorColor.cgColor
    }

    required init?(coder: NSCoder) { fatalError() }

    override var acceptsFirstResponder: Bool { true }

    func summon() {
        NSLog("ipinyin-bar: hotkey 触发，唤出输入条")
        previousApp = NSWorkspace.shared.frontmostApplication
        engine.reset()
        render(preedit: "", candidates: [], hint: "⌃⌥空格已在输入 · Esc 关闭")
        if let screen = NSScreen.main {
            let x = screen.frame.midX - frame.width / 2
            let y = screen.frame.maxY - 120
            setFrameOrigin(NSPoint(x: x, y: y))
        }
        NSApp.activate(ignoringOtherApps: true)
        (window as? NSPanel)?.makeKeyAndOrderFront(nil)
        window?.makeFirstResponder(self)
    }

    private func dismiss() {
        panel?.orderOut(nil)
        previousApp?.activate()
    }

    override func keyDown(with event: NSEvent) {
        guard let keyval = mapKeyval(event) else { return }
        let state = event.modifierFlags.contains(.shift) ? SHIFT_MASK : 0
        guard let reply = engine.event(keyval: keyval, state: state) else { return }

        // Esc：无输入时关闭输入条
        if keyval == 0xFF1B && !reply.visible {
            dismiss()
            return
        }

        if let commit = reply.commit {
            // passthrough（输入中标点）：上屏候选后把标点键也转给前台应用
            commitToPreviousApp(text: commit, forwardKey: reply.handled ? nil : event.keyCode)
            return
        }
        if !reply.handled {
            // 空缓冲时的空格/回车/数字等：原样转给前台应用
            if keyval == 0x20 { commitToPreviousApp(text: "", forwardKey: 49); return }
            if keyval == 0xFF0D || keyval == 0xFF8D { commitToPreviousApp(text: "", forwardKey: 36); return }
        }
        let hint = reply.visible ? "" : "（已上屏）"
        render(preedit: reply.preedit, candidates: reply.candidates, hint: hint)
    }

    private func commitToPreviousApp(text: String, forwardKey: CGKeyCode?) {
        panel?.orderOut(nil)
        previousApp?.activate()
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.12) {
            if !text.isEmpty { TypeInserter.type(text) }
            if let key = forwardKey { TypeInserter.pressKey(key) }
        }
        engine.reset()
        render(preedit: "", candidates: [], hint: "")
        // 重新唤起交给用户：条已隐藏，下次 ⌃⌥空格 再来
    }

    private func render(preedit: String, candidates: [String], hint: String) {
        var parts: [String] = []
        if !preedit.isEmpty { parts.append(preedit) }
        for (i, c) in candidates.prefix(9).enumerated() {
            parts.append("\(i + 1).\(c)")
        }
        if parts.isEmpty { parts.append(hint) }
        label.stringValue = parts.joined(separator: "   ")
    }
}

// ---- 应用壳 -----------------------------------------------------------------

enum BarApp {
    private static var hotKeyRef: EventHotKeyRef?
    private static var summoned: (() -> Void)?

    static func run() {
        let app = NSApplication.shared
        app.setActivationPolicy(.accessory)
        EngineProcess.ensureRunning()   // 预热引擎桥，首次按键不吃冷启动

        // 辅助功能权限：CGEvent 注入所必需；没有就弹系统授权页
        let trusted = AXIsProcessTrustedWithOptions(
            [kAXTrustedCheckOptionPrompt.takeUnretainedValue(): true] as CFDictionary)
        if !trusted {
            NSLog("ipinyin-bar: 等待辅助功能授权")
        }

        let panel = BarPanel(contentRect: NSRect(x: 0, y: 0, width: 620, height: 34),
                             styleMask: [.borderless],
                             backing: .buffered, defer: false)
        panel.level = .floating
        panel.isOpaque = false
        panel.hasShadow = true
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        let view = BarView(panel: panel)
        panel.contentView = view

        summoned = { view.summon() }
        installHotKey()
        NSLog("ipinyin-bar: 就绪（⌃⌥空格 唤出）")
        app.run()
    }

    private static func installHotKey() {
        var eventType = EventTypeSpec(eventClass: OSType(kEventClassKeyboard),
                                      eventKind: UInt32(kEventHotKeyPressed))
        InstallEventHandler(GetApplicationEventTarget(), { _, _, _ -> OSStatus in
            DispatchQueue.main.async { BarApp.summoned?() }
            return noErr
        }, 1, &eventType, nil, nil)
        // ⌃⌥空格
        let hkStatus = RegisterEventHotKey(UInt32(kVK_Space), UInt32(controlKey | optionKey),
                            EventHotKeyID(signature: OSType(0x4950494E), id: 1),
                            GetApplicationEventTarget(), 0, &hotKeyRef)
        NSLog("ipinyin-bar: hotkey 注册 status=%d", hkStatus)
    }
}
