import AVFoundation
import Foundation

struct Turn: Decodable {
    let text: String
    let locale: String
    let rate: Float
    let pitch_offset: Float
}

let speaker = AVSpeechSynthesizer()

func render(_ turn: Turn) -> [String: Any] {
    guard let voice = AVSpeechSynthesisVoice(language: turn.locale) else {
        return ["error": "voice unavailable"]
    }
    let utterance = AVSpeechUtterance(string: turn.text)
    utterance.voice = voice
    utterance.rate = AVSpeechUtteranceDefaultSpeechRate * turn.rate / 168
    utterance.pitchMultiplier = pow(2, turn.pitch_offset / 12)

    var finished = false
    var samples = Data()
    var rate = 0
    let deadline = Date().addingTimeInterval(15)
    speaker.write(utterance) { buffer in
        guard let pcm = buffer as? AVAudioPCMBuffer else { return }
        let length = Int(pcm.frameLength)
        if length == 0 {
            DispatchQueue.main.async { finished = true }
            return
        }
        guard pcm.format.channelCount == 1 else { return }
        let chunk: Data
        if let channel = pcm.int16ChannelData?.pointee {
            chunk = Data(bytes: channel, count: length * MemoryLayout<Int16>.size)
        } else if let channel = pcm.floatChannelData?.pointee {
            let converted = (0..<length).map { index in
                Int16(max(-1, min(1, channel[index])) * 32767)
            }
            chunk = converted.withUnsafeBytes { Data($0) }
        } else {
            return
        }
        let sampleRate = Int(pcm.format.sampleRate)
        DispatchQueue.main.async {
            samples.append(chunk)
            rate = sampleRate
        }
    }
    while !finished && Date() < deadline {
        RunLoop.current.run(until: Date().addingTimeInterval(0.01))
    }
    guard finished, rate > 0, !samples.isEmpty else {
        speaker.stopSpeaking(at: .immediate)
        return ["error": "audio unavailable"]
    }
    return ["rate": rate, "pcm": samples.base64EncodedString()]
}

while let line = readLine() {
    let turn = try? JSONDecoder().decode(Turn.self, from: Data(line.utf8))
    let reply = turn.map(render) ?? ["error": "invalid turn"]
    let payload = try JSONSerialization.data(withJSONObject: reply)
    print(String(decoding: payload, as: UTF8.self))
    fflush(stdout)
}
