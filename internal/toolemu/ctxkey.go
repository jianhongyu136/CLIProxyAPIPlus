package toolemu

import "context"

type ctxKey int

const (
	keyFolded ctxKey = iota + 1
	keyWantsStream
)

// MarkFolded annotates the context to indicate toolemu has folded the request,
// remembering whether the downstream client originally requested streaming.
func MarkFolded(ctx context.Context, downstreamStream bool) context.Context {
	ctx = context.WithValue(ctx, keyFolded, true)
	ctx = context.WithValue(ctx, keyWantsStream, downstreamStream)
	return ctx
}

// IsFolded reports whether MarkFolded has been applied to ctx.
func IsFolded(ctx context.Context) bool {
	v, _ := ctx.Value(keyFolded).(bool)
	return v
}

// WantsStream returns the downstream-stream preference captured by MarkFolded.
func WantsStream(ctx context.Context) bool {
	v, _ := ctx.Value(keyWantsStream).(bool)
	return v
}
