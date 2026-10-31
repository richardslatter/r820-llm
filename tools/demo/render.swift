// Render a recorded token stream (record.py output) to raw RGBA frames on stdout, in real time.
//   swiftc -O render.swift -o render
//   ./render rec.json 30 | ffmpeg -f rawvideo -pix_fmt rgba -s 1920x1080 -r 30 -i - -c:v libx264 -crf 20 -pix_fmt yuv420p demo.mp4
//   ./render rec.json 30 <seconds>   writes frame.png at that time instead
import AppKit
import CoreText
import Foundation

let W = 1920, H = 1080
let args = CommandLine.arguments
let rec = try! JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: args[1]))) as! [String: Any]
let fps = args.count > 2 ? Double(args[2])! : 30
let pngAt = args.count > 3 ? Double(args[3]) : nil

struct Ev { let t: Double; let k: Character; let s: String }
let evs = (rec["events"] as! [[String: Any]]).map { Ev(t: $0["t"] as! Double, k: ($0["k"] as! String).first!, s: $0["s"] as! String) }
let prompt = rec["prompt"] as! String
let endT = rec["end"] as! Double
let timings = rec["timings"] as? [String: Any]
let lead = 2.0, tail = 6.0          // seconds of prompt before the request, and of the summary after
let total = lead + endT + tail

func rgb(_ h: UInt32) -> CGColor {
    CGColor(red: CGFloat((h >> 16) & 255) / 255, green: CGFloat((h >> 8) & 255) / 255, blue: CGFloat(h & 255) / 255, alpha: 1)
}
let bg = rgb(0x0f1115), panel = rgb(0x171a21), border = rgb(0x2a2f3a)
let cText = rgb(0xe6e8ee), cDim = rgb(0x8b93a6), cThink = rgb(0x9aa3b5), cCode = rgb(0x9fd3ff), cFence = rgb(0x5c6577)
let cAccent = rgb(0x7ee08a), cUser = rgb(0xffcf7a), cLabel = rgb(0xc7a6ff)

let fontSize: CGFloat = 21
let mono = CTFontCreateWithName("Menlo" as CFString, fontSize, nil)
let monoB = CTFontCreateWithName("Menlo-Bold" as CFString, fontSize, nil)
let monoI = CTFontCreateWithName("Menlo-Italic" as CFString, fontSize, nil)
let ui = CTFontCreateWithName("Helvetica Neue" as CFString, 24, nil)
let uiB = CTFontCreateWithName("HelveticaNeue-Bold" as CFString, 26, nil)
let uiBig = CTFontCreateWithName("HelveticaNeue-Bold" as CFString, 44, nil)
var adv = CGSize.zero
var glyph = CTFontGetGlyphWithName(mono, "M" as CFString)
CTFontGetAdvancesForGlyphs(mono, .horizontal, &glyph, &adv, 1)
let charW = adv.width
let lineH = (fontSize * 1.42).rounded()

let marginX: CGFloat = 56
let cols = Int((CGFloat(W) - 2 * marginX - 24) / charW)

func draw(_ ctx: CGContext, _ s: String, _ font: CTFont, _ color: CGColor, _ x: CGFloat, _ yTop: CGFloat) {
    let attr = NSAttributedString(string: s, attributes: [.font: font, .foregroundColor: color])
    let line = CTLineCreateWithAttributedString(attr)
    ctx.textPosition = CGPoint(x: x, y: CGFloat(H) - yTop)
    CTLineDraw(line, ctx)
}
func width(_ s: String, _ font: CTFont) -> CGFloat {
    let line = CTLineCreateWithAttributedString(NSAttributedString(string: s, attributes: [.font: font]))
    return CGFloat(CTLineGetTypographicBounds(line, nil, nil, nil))
}

// style per visual line run: 0 text, 1 think, 2 code, 3 fence, 4 label
struct VLine { var runs: [(String, Int)] }

func layout(upTo t: Double) -> (lines: [VLine], nR: Int, nC: Int) {
    var think = "", answer = ""
    var nR = 0, nC = 0
    for e in evs where e.t <= t {
        if e.k == "r" { think += e.s; nR += 1 } else { answer += e.s; nC += 1 }
    }
    var out: [VLine] = []
    func wrap(_ text: String, _ style: Int) {
        var chars = Array(text)
        if chars.isEmpty { out.append(VLine(runs: [("", style)])); return }
        while !chars.isEmpty {
            var n = min(cols, chars.count)
            if n < chars.count, let sp = chars[0..<n].lastIndex(of: " "), sp > cols / 2 { n = sp + 1 }
            out.append(VLine(runs: [(String(chars[0..<n]), style)]))
            chars.removeFirst(n)
        }
    }
    if !think.isEmpty {
        out.append(VLine(runs: [("Thinking", 4)]))
        for l in think.trimmingCharacters(in: .newlines).split(separator: "\n", omittingEmptySubsequences: false) { wrap(String(l), 1) }
        out.append(VLine(runs: [("", 0)]))
    }
    if !answer.isEmpty {
        out.append(VLine(runs: [("Answer", 4)]))
        var inCode = false
        for l in answer.trimmingCharacters(in: .newlines).split(separator: "\n", omittingEmptySubsequences: false) {
            let s = String(l)
            if s.trimmingCharacters(in: .whitespaces).hasPrefix("```") { inCode.toggle(); wrap(s, 3); continue }
            wrap(s, inCode ? 2 : 0)
        }
    }
    return (out, nR, nC)
}

// tokens per second over the last 3 s of chunks (one chunk per token)
func rate(at t: Double) -> Double {
    let w = 3.0
    let first = evs.first?.t ?? 0
    let from = max(first, t - w)
    if t - from < 0.5 { return 0 }
    let n = evs.filter { $0.t > from && $0.t <= t }.count
    return Double(n) / (t - from)
}

let cs = CGColorSpaceCreateDeviceRGB()
let ctx = CGContext(data: nil, width: W, height: H, bitsPerComponent: 8, bytesPerRow: W * 4, space: cs,
                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
let stdout = FileHandle.standardOutput
let nFrames = Int((total * fps).rounded(.up))

func frame(_ ft: Double) {
    let t = ft - lead                      // time since the request was sent
    ctx.setFillColor(bg); ctx.fill(CGRect(x: 0, y: 0, width: W, height: H))

    // header
    draw(ctx, "DeepSeek V4.1 Flash  ·  552B MoE", uiB, cText, marginX, 52)
    draw(ctx, "Dell PowerEdge R820 (2012)  ·  4× Xeon E5-4640, AVX1 only  ·  decode on the CPU, P2200 for prompts  ·  DSpark draft", ui, cDim, marginX, 88)
    let rt = "real-time recording"
    draw(ctx, rt, ui, cAccent, CGFloat(W) - marginX - width(rt, ui), 52)
    ctx.setFillColor(border); ctx.fill(CGRect(x: 0, y: H - 112, width: W, height: 1))

    // prompt
    let pTop: CGFloat = 150
    draw(ctx, "You", uiB, cUser, marginX, pTop)
    var py = pTop + 38
    var pl = Array(prompt)
    while !pl.isEmpty {
        var n = min(cols + 8, pl.count)
        if n < pl.count, let sp = pl[0..<n].lastIndex(of: " ") { n = sp + 1 }
        draw(ctx, String(pl[0..<n]), ui, cText, marginX, py)
        pl.removeFirst(n); py += 34
    }

    // response panel
    let boxTop = py + 18, boxBottom = CGFloat(H) - 96
    ctx.setFillColor(panel)
    ctx.fill(CGRect(x: marginX - 20, y: CGFloat(H) - boxBottom, width: CGFloat(W) - 2 * marginX + 40, height: boxBottom - boxTop))
    let (lines, nR, nC) = layout(upTo: max(t, -1))
    let maxLines = Int((boxBottom - boxTop - 28) / lineH)
    let shown = lines.suffix(maxLines)
    var y = boxTop + 14 + fontSize
    for l in shown {
        var x = marginX
        for (s, st) in l.runs {
            let (f, c): (CTFont, CGColor) = st == 1 ? (monoI, cThink) : st == 2 ? (mono, cCode) : st == 3 ? (mono, cFence) : st == 4 ? (monoB, cLabel) : (mono, cText)
            draw(ctx, s, f, c, x, y)
            x += CGFloat(s.count) * charW
        }
        y += lineH
    }
    if t >= 0 && nR + nC == 0 {
        draw(ctx, "processing prompt…", monoI, cDim, marginX, boxTop + 14 + fontSize)
    }

    // status bar
    let sy = CGFloat(H) - 40
    let el = max(0, min(t, endT))
    let phase = t < 0 ? "ready" : (t >= endT ? "done" : (nC > 0 ? "answering" : (nR > 0 ? "thinking" : "prompt")))
    var status = String(format: "%5.1f s   ·   %@   ·   %d tokens", el, phase, nR + nC)
    if t > 0 && t < endT && nR + nC > 0 { status += String(format: "   ·   %.1f tok/s now", rate(at: t)) }
    if t >= endT {
        func phaseRate(_ k: Character) -> String {
            let ts = evs.filter { $0.k == k }.map { $0.t }
            guard ts.count > 1 else { return "" }
            return String(format: "%d tok at %.1f tok/s", ts.count, Double(ts.count - 1) / (ts.last! - ts.first!))
        }
        status = String(format: "%.1f s   ·   thinking %@   ·   answer %@", endT, phaseRate("r"), phaseRate("c"))
    }
    draw(ctx, status, mono, cDim, marginX, sy)
    if t >= endT, let tm = timings {
        let gen = tm["predicted_per_second"] as! Double, pn = tm["predicted_n"] as! Int
        let s = String(format: "%d tokens at %.1f tok/s", pn, gen)
        draw(ctx, s, uiB, cAccent, CGFloat(W) - marginX - width(s, uiB), sy + 2)
    }
}

if let at = pngAt {
    frame(at)
    let img = ctx.makeImage()!
    let rep = NSBitmapImageRep(cgImage: img)
    try! rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: "frame.png"))
    exit(0)
}
for i in 0..<nFrames {
    frame(Double(i) / fps)
    let p = ctx.data!.assumingMemoryBound(to: UInt8.self)
    stdout.write(Data(bytesNoCopy: p, count: W * H * 4, deallocator: .none))
}
