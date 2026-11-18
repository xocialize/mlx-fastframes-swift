import Darwin
import Foundation
import MLX

struct Args {
    let mode: String
    private var kv: [String: String] = [:]

    init(_ argv: [String]) {
        mode = argv.count > 1 ? argv[1] : "help"
        var i = 2
        while i < argv.count {
            let k = argv[i]
            if k.hasPrefix("--") {
                if i + 1 < argv.count, !argv[i + 1].hasPrefix("--") {
                    kv[String(k.dropFirst(2))] = argv[i + 1]
                    i += 2
                } else {
                    kv[String(k.dropFirst(2))] = "1"
                    i += 1
                }
            } else {
                i += 1
            }
        }
    }

    func str(_ k: String) -> String? { kv[k] }
    func req(_ k: String) -> String {
        guard let v = kv[k] else { fail("missing --\(k)") }
        return v
    }
    func url(_ k: String) -> URL { URL(fileURLWithPath: req(k)) }
    func int(_ k: String, _ d: Int) -> Int { kv[k].flatMap(Int.init) ?? d }
    func double(_ k: String) -> Double? { kv[k].flatMap(Double.init) }
}

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write(("ff-smoke: " + msg + "\n").data(using: .utf8)!)
    exit(2)
}

/// Relative L2 error and cosine in float64 on the host.
func compare(_ ref: MLXArray, _ got: MLXArray) -> (rel: Double, cos: Double) {
    let a = ref.asType(.float32).flattened().asArray(Float.self)
    let b = got.asType(.float32).flattened().asArray(Float.self)
    precondition(a.count == b.count, "size mismatch \(a.count) vs \(b.count)")
    var d2 = 0.0, a2 = 0.0, b2 = 0.0, ab = 0.0
    for i in 0..<a.count {
        let x = Double(a[i]), y = Double(b[i])
        d2 += (x - y) * (x - y); a2 += x * x; b2 += y * y; ab += x * y
    }
    return ((d2 / a2).squareRoot(), ab / (a2.squareRoot() * b2.squareRoot()))
}

/// Per-frame PSNR over uint8 `[N, H, W, 3]`.
func psnr(_ ref: MLXArray, _ got: MLXArray) -> [Double] {
    (0..<ref.dim(0)).map { i in
        let d = ref[i].asType(.float32) - got[i].asType(.float32)
        let mse = Double((d * d).mean().item(Float.self))
        return 10 * log10(255 * 255 / max(mse, 1e-12))
    }
}

/// Current `phys_footprint` (bytes).
func currentFootprint() -> Int {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return kr == KERN_SUCCESS ? Int(info.phys_footprint) : -1
}

/// Peak `phys_footprint` of this process so far (bytes) — the real working-set peak.
func peakFootprint() -> Int {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return kr == KERN_SUCCESS ? Int(info.ledger_phys_footprint_peak) : -1
}

func gb(_ b: Int) -> String { String(format: "%.2f GB", Double(b) / 1e9) }

func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e9 }

