<template>
  <div class="d-sm-flex">
    <div class="ml-1">{{ name }}</div>
    <textarea class="form-control" v-bind:class="{ 'is-invalid': option.params.hasError }"
      :value="value" @input="changeValue($event.target.value)" />
  </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';
import OptionParams from '@/baklava/OptionParams';

@Component
export default class TextArea extends Vue {
  @Prop() value!: string;
  @Prop() name!: string;
  @Prop() option!: {
    params?: OptionParams;
  };

  private changeValue(value: string) {
    this.$emit('input', value);
  }
}
</script>

<style scoped>
textarea {
   font-size: 0.5rem;
}
</style>
