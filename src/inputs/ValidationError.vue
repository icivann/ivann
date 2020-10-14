<template>
  <div class="ValidateError">
  <span
    class="text-sm small text-danger ml-1">
    {{ text }}
  </span>
  </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';

@Component
export default class ValidationError extends Vue {
  @Prop({ required: true }) readonly text!: string;
}
</script>
