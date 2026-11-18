import Testing
@testable import FastFramesMLX

@Suite struct ScheduleTests {
    @Test func flowMatchSigmasMatchUpstreamTimesteps() {
        let s = FlowMatch()
        // shift-5 FlowMatch: t = 1000 ⇒ σ = 1; the FF timesteps sit on grid points
        #expect(s.sigma(1000) == 1)
        #expect(abs(s.sigma(833.3333129882812) - 0.8333333) < 1e-6)
        #expect(abs(s.sigma(625) - 0.625) < 1e-6)
    }

    @Test func geometry() {
        #expect(FF.frameTokens == 1560)
        #expect(FF.numFrames * 4 - FF.droppedLeadingFrames == 81)
    }
}
