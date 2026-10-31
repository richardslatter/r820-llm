// Render a recorded token stream (record.py output) to raw RGBA frames on stdout, in real time.
//   swiftc -O render.swift -o render
//   ./render rec.json 30 | ffmpeg -f rawvideo -pix_fmt rgba -s 1920x1080 -r 30 -i - -c:v libx264 -crf 20 -pix_fmt yuv420p demo.mp4
//   ./render --vertical rec.json 30 | ffmpeg ... -s 1080x1920 ...   portrait, with a large live tok/s readout and graph
//   ./render [--vertical] rec.json 30 <seconds>   writes frame.png at that time instead
import AppKit
import CoreText
import Foundation

var args = Array(CommandLine.arguments.dropFirst())
let vertical = args.contains("--vertical")
args.removeAll { $0 == "--vertical" }
let W = vertical ? 1080 : 1920, H = vertical ? 1920 : 1080
let rec = try! JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: args[0]))) as! [String: Any]
let fps = args.count > 1 ? Double(args[1])! : 30
let pngAt = args.count > 2 ? Double(args[2]) : nil

struct Ev { let t: Double; let k: Character; let s: String }
let evs = (rec["events"] as! [[String: Any]]).map { Ev(t: $0["t"] as! Double, k: ($0["k"] as! String).first!, s: $0["s"] as! String) }
let prompt = rec["prompt"] as! String
let endT = rec["end"] as! Double
let timings = rec["timings"] as? [String: Any]
let lead = 2.0, tail = 6.0          // seconds of prompt before the request, and of the summary after
let total = lead + endT + tail

func rgb(_ h: UInt32, _ a: CGFloat = 1) -> CGColor {
    CGColor(red: CGFloat((h >> 16) & 255) / 255, green: CGFloat((h >> 8) & 255) / 255, blue: CGFloat(h & 255) / 255, alpha: a)
}
let bg = rgb(0x0f1115), panel = rgb(0x171a21), border = rgb(0x2a2f3a)
let cText = rgb(0xe6e8ee), cDim = rgb(0x8b93a6), cThink = rgb(0x9aa3b5), cCode = rgb(0x9fd3ff), cFence = rgb(0x5c6577)
let cAccent = rgb(0x7ee08a), cUser = rgb(0xffcf7a), cLabel = rgb(0xc7a6ff)

let fontSize: CGFloat = vertical ? 23 : 21
let mono = CTFontCreateWithName("Menlo" as CFString, fontSize, nil)
let monoB = CTFontCreateWithName("Menlo-Bold" as CFString, fontSize, nil)
let monoI = CTFontCreateWithName("Menlo-Italic" as CFString, fontSize, nil)
let ui = CTFontCreateWithName("Helvetica Neue" as CFString, 24, nil)
let uiB = CTFontCreateWithName("HelveticaNeue-Bold" as CFString, 26, nil)
var adv = CGSize.zero
var glyph = CTFontGetGlyphWithName(mono, "M" as CFString)
CTFontGetAdvancesForGlyphs(mono, .horizontal, &glyph, &adv, 1)
let charW = adv.width
let lineH = (fontSize * 1.42).rounded()

let marginX: CGFloat = vertical ? 44 : 56
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

// count and rate of the reasoning ("r") or answer ("c") chunks
func phaseStats(_ k: Character) -> (n: Int, rate: Double)? {
    let ts = evs.filter { $0.k == k }.map { $0.t }
    guard ts.count > 1 else { return nil }
    return (ts.count, Double(ts.count - 1) / (ts.last! - ts.first!))
}
func phaseRate(_ k: Character) -> String {
    guard let p = phaseStats(k) else { return "" }
    return String(format: "%d tok at %.1f tok/s", p.n, p.rate)
}

func phase(_ t: Double, _ nR: Int, _ nC: Int) -> String {
    t < 0 ? "ready" : (t >= endT ? "done" : (nC > 0 ? "answering" : (nR > 0 ? "thinking" : "prompt")))
}

// draw the visible tail of the response between yTop and yBottom
func drawResponse(_ ctx: CGContext, _ t: Double, _ boxTop: CGFloat, _ boxBottom: CGFloat) -> (nR: Int, nC: Int) {
    ctx.setFillColor(panel)
    ctx.fill(CGRect(x: marginX - 20, y: CGFloat(H) - boxBottom, width: CGFloat(W) - 2 * marginX + 40, height: boxBottom - boxTop))
    let (lines, nR, nC) = layout(upTo: max(t, -1))
    let maxLines = Int((boxBottom - boxTop - 28) / lineH)
    var y = boxTop + 14 + fontSize
    for l in lines.suffix(maxLines) {
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
    return (nR, nC)
}

let cs = CGColorSpaceCreateDeviceRGB()
let ctx = CGContext(data: nil, width: W, height: H, bitsPerComponent: 8, bytesPerRow: W * 4, space: cs,
                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
let stdout = FileHandle.standardOutput
let nFrames = Int((total * fps).rounded(.up))

func frameLandscape(_ ft: Double) {
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

    let (nR, nC) = drawResponse(ctx, t, py + 18, CGFloat(H) - 96)

    // status bar
    let sy = CGFloat(H) - 40
    let el = max(0, min(t, endT))
    var status = String(format: "%5.1f s   ·   %@   ·   %d tokens", el, phase(t, nR, nC), nR + nC)
    if t > 0 && t < endT && nR + nC > 0 { status += String(format: "   ·   %.1f tok/s now", rate(at: t)) }
    if t >= endT {
        status = String(format: "%.1f s   ·   thinking %@   ·   answer %@", endT, phaseRate("r"), phaseRate("c"))
    }
    draw(ctx, status, mono, cDim, marginX, sy)
    if t >= endT, let tm = timings {
        let gen = tm["predicted_per_second"] as! Double, pn = tm["predicted_n"] as! Int
        let s = String(format: "%d tokens at %.1f tok/s", pn, gen)
        draw(ctx, s, uiB, cAccent, CGFloat(W) - marginX - width(s, uiB), sy + 2)
    }
}

// portrait: the speed panel on top (live tok/s, a graph of it so far), then the prompt and the response
let vTitle = CTFontCreateWithName("HelveticaNeue-Bold" as CFString, 40, nil)
let vSub = CTFontCreateWithName("Helvetica Neue" as CFString, 27, nil)
let vPhase = CTFontCreateWithName("HelveticaNeue-Bold" as CFString, 32, nil)
let vHuge = CTFontCreateWithName("HelveticaNeue-Bold" as CFString, 190, nil)
let vUnit = CTFontCreateWithName("HelveticaNeue-Bold" as CFString, 54, nil)
let vInfo = CTFontCreateWithName("Menlo" as CFString, 27, nil)
let vSmall = CTFontCreateWithName("Menlo" as CFString, 20, nil)
let vPrompt = CTFontCreateWithName("Helvetica Neue" as CFString, 31, nil)

// the rate graph: one sample every 0.25 s from the first token to the end
let firstT = evs.first?.t ?? 0
let firstAnswerT = evs.first { $0.k == "c" }?.t ?? endT
let series: [(t: Double, r: Double)] = stride(from: firstT + 0.5, through: endT, by: 0.25).map { ($0, rate(at: $0)) }
let graphMax = max(5, ((series.map { $0.r }.max() ?? 10) / 5).rounded(.up) * 5)

func wrapWords(_ text: String, _ font: CTFont, _ maxW: CGFloat) -> [String] {
    var lines: [String] = [], cur = ""
    for w in text.split(separator: " ") {
        let cand = cur.isEmpty ? String(w) : cur + " " + w
        if width(cand, font) > maxW && !cur.isEmpty { lines.append(cur); cur = String(w) } else { cur = cand }
    }
    if !cur.isEmpty { lines.append(cur) }
    return lines
}

func frameVertical(_ ft: Double) {
    let t = ft - lead
    let w = CGFloat(W), h = CGFloat(H)
    ctx.setFillColor(bg); ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))

    // header
    draw(ctx, "DeepSeek V4.1 Flash  ·  552B MoE", vTitle, cText, marginX, 82)
    draw(ctx, "Dell PowerEdge R820 (2012)  ·  4× Xeon E5-4640", vSub, cDim, marginX, 128)
    draw(ctx, "AVX1 only  ·  decode on the CPU  ·  DSpark draft", vSub, cDim, marginX, 166)
    let rt = "● real time"
    draw(ctx, rt, vSub, cAccent, w - marginX - width(rt, vSub), 166)

    // speed panel
    let pTop: CGFloat = 200, pBottom: CGFloat = 720
    ctx.setFillColor(panel)
    ctx.addPath(CGPath(roundedRect: CGRect(x: marginX - 20, y: h - pBottom, width: w - 2 * marginX + 40, height: pBottom - pTop),
                       cornerWidth: 18, cornerHeight: 18, transform: nil))
    ctx.fillPath()

    var nR = 0, nC = 0
    for e in evs where e.t <= t { if e.k == "r" { nR += 1 } else { nC += 1 } }
    let ph = phase(t, nR, nC)
    let done = t >= endT
    let (phText, phColor): (String, CGColor) = ph == "thinking" ? ("THINKING", cThink) : ph == "answering" ? ("WRITING THE ANSWER", cCode)
        : ph == "done" ? ("DONE", cAccent) : ph == "prompt" ? ("READING THE PROMPT", cDim) : ("READY", cDim)
    draw(ctx, phText, vPhase, phColor, marginX, pTop + 58)
    let el = String(format: "%.1f s", max(0, min(t, endT)))
    draw(ctx, el, vInfo, cDim, w - marginX - width(el, vInfo), pTop + 56)

    // the big number: the last 3 s while running (updated 4 times a second, so it can be read), the
    // server's average at the end
    var big = "–"
    if done, let tm = timings { big = String(format: "%.1f", tm["predicted_per_second"] as! Double) }
    else if t > 0 && nR + nC > 0 { big = String(format: "%.1f", rate(at: max(firstT + 0.5, (t*4).rounded(.down)/4))) }
    let bigY = pTop + 250
    draw(ctx, big, vHuge, done ? cAccent : cText, marginX - 6, bigY)
    let unit = done ? "tok/s average" : "tok/s"
    draw(ctx, unit, vUnit, cDim, marginX + width(big, vHuge) + 18, bigY)

    // one line under the number
    var info = ""
    if done, let r = phaseStats("r"), let c = phaseStats("c") {
        let n = timings?["predicted_n"] as? Int ?? nR + nC
        info = String(format: "%d tokens  ·  thinking %.1f  ·  answer %.1f", n, r.rate, c.rate)
    } else if t > 0 && nR + nC > 0 {
        info = String(format: "%d tokens  ·  %.1f tok/s so far", nR + nC, Double(nR + nC - 1) / max(0.5, t - firstT))
    } else if t >= 0 {
        info = "processing the prompt…"
    }
    draw(ctx, info, vInfo, cText, marginX, bigY + 62)

    // graph of tok/s over time, thinking in grey, the answer in green
    let gl = marginX, gr = w - marginX, gTop = bigY + 110, gBottom = pBottom - 34
    let tSpan = max(1, endT - firstT)
    func gx(_ ts: Double) -> CGFloat { gl + CGFloat((ts - firstT) / tSpan) * (gr - gl) }
    func gy(_ r: Double) -> CGFloat { h - (gBottom - CGFloat(r / graphMax) * (gBottom - gTop)) }
    ctx.setFillColor(border)
    for v in stride(from: 0.0, through: graphMax, by: 5.0) { ctx.fill(CGRect(x: gl, y: gy(v), width: gr - gl, height: 1)) }
    draw(ctx, String(format: "%.0f", graphMax), vSmall, cDim, gl, gBottom - (gBottom - gTop) - 6)
    let shown = series.filter { $0.t <= t }
    if shown.count > 1 {
        for (from, to, color) in [(firstT, firstAnswerT, cThink), (firstAnswerT, endT + 1, cAccent)] {
            let seg = shown.filter { $0.t >= from - 0.25 && $0.t <= to }
            guard seg.count > 1 else { continue }
            let area = CGMutablePath()
            area.move(to: CGPoint(x: gx(seg[0].t), y: gy(0)))
            for p in seg { area.addLine(to: CGPoint(x: gx(p.t), y: gy(p.r))) }
            area.addLine(to: CGPoint(x: gx(seg.last!.t), y: gy(0)))
            ctx.addPath(area); ctx.setFillColor(color.copy(alpha: 0.18)!); ctx.fillPath()
            let line = CGMutablePath()
            line.move(to: CGPoint(x: gx(seg[0].t), y: gy(seg[0].r)))
            for p in seg.dropFirst() { line.addLine(to: CGPoint(x: gx(p.t), y: gy(p.r))) }
            ctx.addPath(line); ctx.setStrokeColor(color); ctx.setLineWidth(3); ctx.setLineJoin(.round); ctx.strokePath()
        }
    }
    draw(ctx, "tok/s over time", vSmall, cDim, gl, pBottom - 8)
    let legend = "thinking   answer"
    let lx = gr - width(legend, vSmall)
    draw(ctx, "thinking", vSmall, cThink, lx, pBottom - 8)
    draw(ctx, "answer", vSmall, cAccent, lx + width("thinking   ", vSmall), pBottom - 8)

    // prompt
    var py: CGFloat = pBottom + 62
    draw(ctx, "You", vPhase, cUser, marginX, py)
    py += 46
    for l in wrapWords(prompt, vPrompt, w - 2 * marginX) { draw(ctx, l, vPrompt, cText, marginX, py); py += 40 }

    _ = drawResponse(ctx, t, py + 16, h - 40)
}

func frame(_ ft: Double) { vertical ? frameVertical(ft) : frameLandscape(ft) }

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
