module github.com/gomlx/go-xla

go 1.24.3

toolchain go1.24.6

require (
	github.com/janpfeifer/go-benchmarks v0.1.1
	github.com/pkg/errors v0.9.1
	github.com/x448/float16 v0.8.4
	golang.org/x/term v0.38.0
	google.golang.org/protobuf v1.36.10
	k8s.io/klog/v2 v2.130.1
)

require (
	github.com/dmarkham/enumer v1.6.1 // indirect
	github.com/go-logr/logr v1.4.3 // indirect
	github.com/pascaldekloe/name v1.0.0 // indirect
	github.com/streadway/quantile v0.0.0-20220407130108-4246515d968d // indirect
	golang.org/x/mod v0.29.0 // indirect
	golang.org/x/sync v0.17.0 // indirect
	golang.org/x/sys v0.39.0 // indirect
	golang.org/x/tools v0.38.0 // indirect
)

tool github.com/dmarkham/enumer

replace github.com/janpfeifer/go-benchmarks => github.com/ajroetker/go-benchmarks v0.0.0-20251227191922-60d02b7acb0c
