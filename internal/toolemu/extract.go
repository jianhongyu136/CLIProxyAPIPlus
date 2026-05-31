package toolemu

import (
	"fmt"
	"strings"

	"github.com/tidwall/gjson"
)

// ExtractAssistantText pulls the full assistant text body from an upstream
// non-streaming response, along with the metadata required by emitters.
func ExtractAssistantText(body []byte, shape UpstreamShape) (string, UpstreamMeta, error) {
	switch shape {
	case ShapeOpenAIChat:
		return extractChatText(body)
	case ShapeOpenAIResponses:
		return extractResponsesText(body)
	case ShapeClaudeMessages:
		return extractClaudeText(body)
	case ShapeGeminiGenerateContent:
		return extractGeminiText(body)
	default:
		return "", UpstreamMeta{}, fmt.Errorf("toolemu: unknown shape %d", shape)
	}
}

func extractChatText(body []byte) (string, UpstreamMeta, error) {
	root := gjson.ParseBytes(body)
	text := root.Get("choices.0.message.content").String()
	meta := UpstreamMeta{
		ResponseID:   root.Get("id").String(),
		Model:        root.Get("model").String(),
		Created:      root.Get("created").Int(),
		UsagePayload: []byte(root.Get("usage").Raw),
	}
	return text, meta, nil
}

func extractResponsesText(body []byte) (string, UpstreamMeta, error) {
	root := gjson.ParseBytes(body)
	var sb strings.Builder
	root.Get("output").ForEach(func(_, item gjson.Result) bool {
		if item.Get("type").String() != "message" {
			return true
		}
		item.Get("content").ForEach(func(_, part gjson.Result) bool {
			if part.Get("type").String() == "output_text" {
				sb.WriteString(part.Get("text").String())
			}
			return true
		})
		return true
	})
	created := root.Get("created_at").Int()
	if created == 0 {
		created = root.Get("created").Int()
	}
	meta := UpstreamMeta{
		ResponseID:   root.Get("id").String(),
		Model:        root.Get("model").String(),
		Created:      created,
		UsagePayload: []byte(root.Get("usage").Raw),
	}
	return sb.String(), meta, nil
}

func extractClaudeText(body []byte) (string, UpstreamMeta, error) {
	root := gjson.ParseBytes(body)
	var sb strings.Builder
	root.Get("content").ForEach(func(_, part gjson.Result) bool {
		if part.Get("type").String() == "text" {
			sb.WriteString(part.Get("text").String())
		}
		return true
	})
	meta := UpstreamMeta{
		ResponseID:   root.Get("id").String(),
		Model:        root.Get("model").String(),
		UsagePayload: []byte(root.Get("usage").Raw),
	}
	return sb.String(), meta, nil
}

func extractGeminiText(body []byte) (string, UpstreamMeta, error) {
	root := gjson.ParseBytes(body)
	var sb strings.Builder
	root.Get("candidates.0.content.parts").ForEach(func(_, part gjson.Result) bool {
		if text := part.Get("text").String(); text != "" {
			sb.WriteString(text)
		}
		return true
	})
	meta := UpstreamMeta{
		Model:        root.Get("modelVersion").String(),
		UsagePayload: []byte(root.Get("usageMetadata").Raw),
	}
	return sb.String(), meta, nil
}
