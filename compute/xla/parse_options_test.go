// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package xla

import (
	"slices"
	"testing"
)

func TestParseOptions(t *testing.T) {
	// Test bool option
	opts := map[string]string{
		"foo":    "true",
		"bar":    "false",
		"baz":    "",
		"nofizz": "",
	}

	val, found, err := parseOptions[bool]("foo", opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !found {
		t.Errorf("expected found=true")
	}
	if !val {
		t.Errorf("expected val=true")
	}
	if _, ok := opts["foo"]; ok {
		t.Errorf("expected 'foo' to be removed from opts")
	}

	val, found, err = parseOptions[bool]("bar", opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !found {
		t.Errorf("expected found=true")
	}
	if val {
		t.Errorf("expected val=false")
	}
	if _, ok := opts["bar"]; ok {
		t.Errorf("expected 'bar' to be removed from opts")
	}

	val, found, err = parseOptions[bool]("baz", opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !found {
		t.Errorf("expected found=true")
	}
	if !val {
		t.Errorf("expected val=true")
	}
	if _, ok := opts["baz"]; ok {
		t.Errorf("expected 'baz' to be removed from opts")
	}

	val, found, err = parseOptions[bool]("fizz", opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !found {
		t.Errorf("expected found=true")
	}
	if val {
		t.Errorf("expected val=false")
	}
	if _, ok := opts["nofizz"]; ok {
		t.Errorf("expected 'nofizz' to be removed from opts")
	}

	// Test []int64 option
	opts = map[string]string{
		"devices1": "0;1;2",
		"devices2": "3:4:5",
		"devices3": "6 7 8",
		"devices4": "9",
		"devices5": "",
	}

	valList, found, err := parseOptions[[]int64]("devices1", opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !found {
		t.Errorf("expected found=true")
	}
	if want := []int64{0, 1, 2}; !slices.Equal(valList, want) {
		t.Errorf("got %v, want %v", valList, want)
	}

	valList, found, err = parseOptions[[]int64]("devices2", opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !found {
		t.Errorf("expected found=true")
	}
	if want := []int64{3, 4, 5}; !slices.Equal(valList, want) {
		t.Errorf("got %v, want %v", valList, want)
	}

	valList, found, err = parseOptions[[]int64]("devices3", opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !found {
		t.Errorf("expected found=true")
	}
	if want := []int64{6, 7, 8}; !slices.Equal(valList, want) {
		t.Errorf("got %v, want %v", valList, want)
	}

	valList, found, err = parseOptions[[]int64]("devices4", opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !found {
		t.Errorf("expected found=true")
	}
	if want := []int64{9}; !slices.Equal(valList, want) {
		t.Errorf("got %v, want %v", valList, want)
	}

	valList, found, err = parseOptions[[]int64]("devices5", opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !found {
		t.Errorf("expected found=true")
	}
	if valList != nil {
		t.Errorf("expected valList to be nil, got %v", valList)
	}

	// Test error cases
	opts = map[string]string{
		"bad_bool": "invalid",
		"bad_int":  "1;foo",
	}

	_, _, err = parseOptions[bool]("bad_bool", opts)
	if err == nil {
		t.Errorf("expected error for bad_bool")
	}

	_, _, err = parseOptions[[]int64]("bad_int", opts)
	if err == nil {
		t.Errorf("expected error for bad_int")
	}
}
