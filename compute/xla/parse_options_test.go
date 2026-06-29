// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package xla

import (
	"testing"

	"github.com/stretchr/testify/assert"
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
	assert.NoError(t, err)
	assert.True(t, found)
	assert.True(t, val)
	assert.NotContains(t, opts, "foo")

	val, found, err = parseOptions[bool]("bar", opts)
	assert.NoError(t, err)
	assert.True(t, found)
	assert.False(t, val)
	assert.NotContains(t, opts, "bar")

	val, found, err = parseOptions[bool]("baz", opts)
	assert.NoError(t, err)
	assert.True(t, found)
	assert.True(t, val)
	assert.NotContains(t, opts, "baz")

	val, found, err = parseOptions[bool]("fizz", opts)
	assert.NoError(t, err)
	assert.True(t, found)
	assert.False(t, val)
	assert.NotContains(t, opts, "nofizz")

	// Test []int64 option
	opts = map[string]string{
		"devices1": "0;1;2",
		"devices2": "3:4:5",
		"devices3": "6 7 8",
		"devices4": "9",
		"devices5": "",
	}

	valList, found, err := parseOptions[[]int64]("devices1", opts)
	assert.NoError(t, err)
	assert.True(t, found)
	assert.Equal(t, []int64{0, 1, 2}, valList)

	valList, found, err = parseOptions[[]int64]("devices2", opts)
	assert.NoError(t, err)
	assert.True(t, found)
	assert.Equal(t, []int64{3, 4, 5}, valList)

	valList, found, err = parseOptions[[]int64]("devices3", opts)
	assert.NoError(t, err)
	assert.True(t, found)
	assert.Equal(t, []int64{6, 7, 8}, valList)

	valList, found, err = parseOptions[[]int64]("devices4", opts)
	assert.NoError(t, err)
	assert.True(t, found)
	assert.Equal(t, []int64{9}, valList)

	valList, found, err = parseOptions[[]int64]("devices5", opts)
	assert.NoError(t, err)
	assert.True(t, found)
	assert.Nil(t, valList)

	// Test error cases
	opts = map[string]string{
		"bad_bool": "invalid",
		"bad_int":  "1;foo",
	}

	_, _, err = parseOptions[bool]("bad_bool", opts)
	assert.Error(t, err)

	_, _, err = parseOptions[[]int64]("bad_int", opts)
	assert.Error(t, err)
}
