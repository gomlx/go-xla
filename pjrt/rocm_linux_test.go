//go:build linux && amd64

package pjrt

import "testing"

func TestRocminfoHasDiscreteGPU(t *testing.T) {
	discreteAndAPU := `*******
Agent 1                  
*******
  Name:                    gfx1100                            
  Marketing Name:          AMD Radeon RX 7900 XTX             
  Memory Properties:       
*******
Agent 2                  
*******
  Name:                    gfx1036                            
  Marketing Name:          AMD Ryzen 9 7900X 12-Core Processor
  Memory Properties:       APU
`
	if !rocminfoHasDiscreteGPU(discreteAndAPU) {
		t.Error("rocminfoHasDiscreteGPU() = false, want true (has a discrete GPU)")
	}
}

func TestRocminfoHasDiscreteGPUAPUOnly(t *testing.T) {
	apuOnly := `*******
Agent 1                  
*******
  Name:                    gfx1036                            
  Marketing Name:          AMD Ryzen 9 7900X 12-Core Processor
  Memory Properties:       APU
`
	if rocminfoHasDiscreteGPU(apuOnly) {
		t.Error("rocminfoHasDiscreteGPU() = true, want false (APU only)")
	}
}
