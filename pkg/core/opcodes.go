package core

// Default opcode mapping
var DefaultOpcodeMap = map[string]string{
	"prompt":    "/gnd/prompt",
	"let":       "/gnd/let",
	"select":    "/gnd/select",
	"concat":    "/gnd/concat",
	"lowercase": "/gnd/lowercase",
	"uppercase": "/gnd/uppercase",
	"trim":      "/gnd/trim",
	"print":     "/gnd/print",
	"log":       "/gnd/log",
	"error":     "/gnd/error",
	"warn":      "/gnd/warn",
	"info":      "/gnd/info",
	"debug":     "/gnd/debug",
	"exit":      "/gnd/exit",
	"return":    "/gnd/return",
}
