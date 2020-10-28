<template>
  <div class="d-sm-flex">
    <div class="ml-1">{{ name }}</div>
    <textarea class="form-control" v-bind:class="{ 'is-invalid': value.hasError }"
      :value="value.text" @input="changeValue($event.target.value)" />
  </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';

@Component
export default class TextArea extends Vue {
  /** Value is a tuple of the shape (text, hasError).
   * If `hasError` is set, the TextArea is highlighted.
   */
  @Prop() value!: {
    text: string;
    hasError: boolean;
  };
  @Prop() name!: string;

  private changeValue(value: string) {
    this.$emit('input', { text: value, hasError: this.value.hasError });
  }
}
</script>

<style scoped>
textarea {
   font-size: 0.5rem;
}
</style>
