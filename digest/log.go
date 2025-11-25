package main

import (
	"fmt"
	"os"
	"strings"
)

type LogLevel int

const (
	LogLevelError LogLevel = iota
	LogLevelWarn
	LogLevelInfo
	LogLevelDebug
)

var currentLogLevel = LogLevelInfo

func init() {
	level := strings.ToLower(os.Getenv("DIGEST_LOG_LEVEL"))
	switch level {
	case "debug":
		currentLogLevel = LogLevelDebug
	case "info":
		currentLogLevel = LogLevelInfo
	case "warn":
		currentLogLevel = LogLevelWarn
	case "error":
		currentLogLevel = LogLevelError
	case "silent", "none":
		currentLogLevel = LogLevel(-1)
	}
}

func logDebug(format string, args ...any) {
	if currentLogLevel >= LogLevelDebug {
		fmt.Fprintf(os.Stderr, "[DEBUG] "+format+"\n", args...)
	}
}

func logInfo(format string, args ...any) {
	if currentLogLevel >= LogLevelInfo {
		fmt.Fprintf(os.Stderr, format+"\n", args...)
	}
}

func logWarn(format string, args ...any) {
	if currentLogLevel >= LogLevelWarn {
		fmt.Fprintf(os.Stderr, "[WARN] "+format+"\n", args...)
	}
}

func logError(format string, args ...any) {
	if currentLogLevel >= LogLevelError {
		fmt.Fprintf(os.Stderr, "[ERROR] "+format+"\n", args...)
	}
}
