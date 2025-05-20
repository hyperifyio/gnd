package model

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"io/fs"
	"os"
	"strings"
	"testing"
	"time"
)

type testFS struct {
	files map[string][]byte
}

func (t *testFS) Open(name string) (fs.File, error) {
	if data, ok := t.files[name]; ok {
		return &testFile{data: data}, nil
	}
	return nil, os.ErrNotExist
}

type testFile struct {
	data []byte
	pos  int64
}

func (t *testFile) Read(p []byte) (n int, err error) {
	if t.pos >= int64(len(t.data)) {
		return 0, io.EOF
	}
	n = copy(p, t.data[t.pos:])
	t.pos += int64(n)
	return n, nil
}

func (t *testFile) Close() error {
	return nil
}

func (t *testFile) Stat() (fs.FileInfo, error) {
	return &testFileInfo{size: int64(len(t.data))}, nil
}

type testFileInfo struct {
	size int64
}

func (t *testFileInfo) Name() string       { return "" }
func (t *testFileInfo) Size() int64        { return t.size }
func (t *testFileInfo) Mode() fs.FileMode  { return 0 }
func (t *testFileInfo) ModTime() time.Time { return time.Time{} }
func (t *testFileInfo) IsDir() bool        { return false }
func (t *testFileInfo) Sys() interface{}   { return nil }

func TestNewModelLoader(t *testing.T) {
	// Create a test GGUF file
	header := &GGUFHeader{
		Magic:       0x46554747, // GGUF magic number
		Version:     1,
		TensorCount: 10,
		KVCount:     5,
	}

	var buf bytes.Buffer
	if err := binary.Write(&buf, binary.LittleEndian, header); err != nil {
		t.Fatal(err)
	}

	testFS := &testFS{
		files: map[string][]byte{
			"model.bin": buf.Bytes(),
		},
	}

	loader, err := NewModelLoader(testFS, "model.bin")
	if err != nil {
		t.Fatalf("NewModelLoader failed: %v", err)
	}

	if loader == nil {
		t.Fatal("NewModelLoader returned nil")
	}

	if loader.modelPath != "model.bin" {
		t.Errorf("expected modelPath to be 'model.bin', got %q", loader.modelPath)
	}

	if loader.bufferSize != 1024*1024 {
		t.Errorf("expected bufferSize to be 1MB, got %d", loader.bufferSize)
	}

	if loader.header == nil {
		t.Fatal("expected header to be loaded")
	}

	if loader.header.Magic != 0x46554747 {
		t.Errorf("expected magic number 0x46554747, got 0x%x", loader.header.Magic)
	}
}

func TestNewModelLoaderErrors(t *testing.T) {
	tests := []struct {
		name      string
		fs        fs.FS
		modelPath string
		wantErr   error
	}{
		{
			name:      "nil filesystem",
			fs:        nil,
			modelPath: "model.bin",
			wantErr:   errors.New("filesystem cannot be nil"),
		},
		{
			name:      "empty model path",
			fs:        &testFS{},
			modelPath: "",
			wantErr:   errors.New("model path cannot be empty"),
		},
		{
			name:      "file not found",
			fs:        &testFS{},
			modelPath: "nonexistent.bin",
			wantErr:   ErrModelNotFound,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewModelLoader(tt.fs, tt.modelPath)
			if err == nil {
				t.Fatal("expected error, got nil")
			}
			if err.Error() != tt.wantErr.Error() {
				t.Errorf("expected error %q, got %q", tt.wantErr, err)
			}
		})
	}
}

func TestLoadModel(t *testing.T) {
	testFS := &testFS{
		files: map[string][]byte{
			"model.bin": []byte("test data"),
		},
	}

	loader := &ModelLoader{
		fs:        testFS,
		modelPath: "model.bin",
	}

	file, err := loader.LoadModel()
	if err != nil {
		t.Fatalf("LoadModel failed: %v", err)
	}
	defer file.Close()

	data := make([]byte, 9)
	n, err := file.Read(data)
	if err != nil {
		t.Fatalf("Read failed: %v", err)
	}

	if n != 9 {
		t.Errorf("expected to read 9 bytes, got %d", n)
	}

	if string(data) != "test data" {
		t.Errorf("expected data to be 'test data', got %q", string(data))
	}
}

func TestLoadModelErrors(t *testing.T) {
	loader := &ModelLoader{
		fs:        &testFS{},
		modelPath: "",
	}

	_, err := loader.LoadModel()
	if err != ErrModelNotSet {
		t.Errorf("expected ErrModelNotSet, got %v", err)
	}
}

func TestGetModelSize(t *testing.T) {
	testFS := &testFS{
		files: map[string][]byte{
			"model.bin": []byte("test data"),
		},
	}

	loader := &ModelLoader{
		fs:        testFS,
		modelPath: "model.bin",
	}

	size, err := loader.GetModelSize()
	if err != nil {
		t.Fatalf("GetModelSize failed: %v", err)
	}

	if size != 9 {
		t.Errorf("expected size to be 9, got %d", size)
	}
}

func TestLoadModelStream(t *testing.T) {
	testFS := &testFS{
		files: map[string][]byte{
			"model.bin": []byte("test data"),
		},
	}

	loader := &ModelLoader{
		fs:        testFS,
		modelPath: "model.bin",
	}

	reader, file, err := loader.LoadModelStream()
	if err != nil {
		t.Fatalf("LoadModelStream failed: %v", err)
	}
	defer file.Close()

	data, err := reader.ReadString('\n')
	if err != nil && err != io.EOF {
		t.Fatalf("ReadString failed: %v", err)
	}

	if data != "test data" {
		t.Errorf("expected data to be 'test data', got %q", data)
	}
}

func TestLoadModelStreamErrors(t *testing.T) {
	loader := &ModelLoader{
		fs:        &testFS{},
		modelPath: "",
	}

	_, _, err := loader.LoadModelStream()
	if err != ErrModelNotSet {
		t.Errorf("expected ErrModelNotSet, got %v", err)
	}
}

func TestLoadModelChunk(t *testing.T) {
	reader := bufio.NewReader(strings.NewReader("test data"))
	loader := &ModelLoader{}

	chunk, err := loader.LoadModelChunk(reader, 4)
	if err != nil {
		t.Fatalf("LoadModelChunk failed: %v", err)
	}

	if string(chunk) != "test" {
		t.Errorf("expected chunk to be 'test', got %q", string(chunk))
	}
}

func TestLoadModelChunkErrors(t *testing.T) {
	loader := &ModelLoader{}

	_, err := loader.LoadModelChunk(nil, 4)
	if err != ErrReaderNil {
		t.Errorf("expected ErrReaderNil, got %v", err)
	}
}
