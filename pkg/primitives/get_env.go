package primitives

import "os"

func GetEnv(key, defaultValue string) string {
	var value = os.Getenv(key)
	if value == "" {
		return defaultValue
	}
	return value
}
