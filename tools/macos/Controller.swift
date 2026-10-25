// IMK 输入法控制器：把 NSEvent 翻译成引擎事件，把引擎状态画到屏幕上。
// 协议见 tools/macos_server.py 的 docstring。
import Cocoa
import InputMethodKit

// 与 ipinyin/keys.py 对齐的修饰键位
private let SHIFT_MASK = 1 << 0

// @objc 固定 ObjC 类名：Info.plist 的 InputMethodServerControllerClass 按它解析
@objc(PinyinController)
final class PinyinController: IMKInputController {
    private let engine = EngineClient.shared
    private var candidatesWindow: IMKCandidates?
    private var displayStrings: [NSAttributedString] = []
    private var highlightIndex = 0
    private var currentPage = 0
    private let pageSize = 9
    private var asciiMode = false            // Shift 轻按 = 中/英切换
    private var shiftDownAt: TimeInterval = 0
    private var typedDuringShift = false

    override init(server: IMKServer!, delegate: Any!, client: Any!) {
        super.init(server: server, delegate: delegate, client: client)
        if let server = server {
            // kIMKSingleRowSteppingCandidatePanel = 3：9 列单行，和 Linux 版候选一致
            candidatesWindow = IMKCandidates(server: server, panelType: IMKCandidatePanelType(3))
        }
    }

    // ---- 事件入口 -------------------------------------------------------

    override func handle(_ event: NSEvent!, client: Any!) -> Bool {
        guard let client = client as? IMKTextInput, let event = event else { return false }
        if event.type == .flagsChanged {
            return handleFlagsChanged(event)
        }
        guard event.type == .keyDown else { return false }
        if asciiMode { return false }                 // 英文模式：全部放行

        let mods = event.modifierFlags.intersection(.deviceIndependentFlagsMask)
        if mods.contains(.shift) { typedDuringShift = true }
        // Ctrl/Option/Command 组合键不归输入法管
        guard mods.subtracting([.shift, .capsLock, .function]).isEmpty else { return false }
        guard let keyval = mapKeyval(event) else { return false }
        let state = mods.contains(.shift) ? SHIFT_MASK : 0

        guard let reply = engine.event(keyval: keyval, state: state) else { return false }
        let commit = reply.commit
        if let commit = commit {
            client.insertText(commit as NSString,
                              replacementRange: NSRange(location: NSNotFound, length: NSNotFound))
        }
        refresh(client: client, reply: reply)
        if reply.handled { return true }
        // passthrough（输入中标点）：候选已上屏，按键本身交还应用
        return commit == nil
    }

    // Shift 轻按（无中间按键、<350ms）切中/英
    private func handleFlagsChanged(_ event: NSEvent) -> Bool {
        let mods = event.modifierFlags.intersection(.deviceIndependentFlagsMask)
        if mods.contains(.shift) {
            if shiftDownAt == 0 {
                shiftDownAt = Date.timeIntervalSinceReferenceDate
                typedDuringShift = false
            }
        } else if shiftDownAt != 0 {
            let elapsed = Date.timeIntervalSinceReferenceDate - shiftDownAt
            shiftDownAt = 0
            if elapsed < 0.35 && !typedDuringShift {
                asciiMode.toggle()
            }
        }
        return true
    }

    // ---- 候选窗（IMK 动态派发的非正式协议，选择器必须与 ObjC 侧一致）------

    override func candidates(_ sender: Any!) -> [Any]? {
        return displayStrings
    }

    @objc(selectedCandidate)
    func selectedCandidateImpl() -> Int {
        return highlightIndex
    }

    @objc(candidateSelectionChanged:client:)
    func candidateSelectionChangedImpl(_ candidateString: Any!, client: Any!) {
        if let index = indexOf(candidateString) {
            highlightIndex = index
        }
    }

    @objc(candidateSelected:client:)
    func candidateSelectedImpl(_ candidateString: Any!, client: Any!) {
        guard let client = client as? IMKTextInput,
              let index = indexOf(candidateString),
              let reply = engine.select(index: currentPage * pageSize + index) else { return }
        if let commit = reply.commit {
            client.insertText(commit as NSString,
                              replacementRange: NSRange(location: NSNotFound, length: NSNotFound))
        }
        refresh(client: client, reply: reply)
    }

    private func indexOf(_ candidateString: Any!) -> Int? {
        if let s = candidateString as? String {
            return displayStrings.firstIndex { $0.string == s }
        }
        if let s = candidateString as? NSAttributedString {
            return displayStrings.firstIndex { $0.string == s.string }
        }
        return nil
    }

    // ---- 画布 -----------------------------------------------------------

    private func refresh(client: IMKTextInput, reply: EngineReply) {
        if reply.visible && !reply.preedit.isEmpty {
            let attrs: [NSAttributedString.Key: Any] =
                [.underlineStyle: NSUnderlineStyle.single.rawValue]
            let marked = NSAttributedString(string: reply.preedit, attributes: attrs)
            client.setMarkedText(marked,
                                 selectionRange: NSRange(location: (reply.preedit as NSString).length, length: 0),
                                 replacementRange: NSRange(location: NSNotFound, length: NSNotFound))
        } else {
            client.setMarkedText("",
                                 selectionRange: NSRange(location: 0, length: 0),
                                 replacementRange: NSRange(location: NSNotFound, length: NSNotFound))
        }

        if reply.visible && !reply.candidates.isEmpty {
            displayStrings = reply.candidates.map { NSAttributedString(string: $0) }
            highlightIndex = reply.highlight
            currentPage = reply.page
            candidatesWindow?.show(IMKCandidatesLocationHint(1))   // 跟随插入点上方
            candidatesWindow?.update()
        } else {
            displayStrings = []
            candidatesWindow?.hide()
        }
    }

    // ---- 键值映射（macOS keyCode → IBus keyval，字符键直接取字符）-------

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
}
