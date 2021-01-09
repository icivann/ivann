<template>
  <div class="checkbox" :class="checked !== checkboxValue.UNCHECKED && 'checked'" @click="clicked">
    <span v-if="checked === checkboxValue.CHECKED" class="tick">✓</span>
    <span v-else-if="checked === checkboxValue.HALFCHECKED" class="dash">-</span>
  </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';
import CheckboxValue from '@/baklava/CheckboxValue';

@Component({})
export default class CheckboxInput extends Vue {
  @Prop({ required: true }) checked!: CheckboxValue;

  private checkboxValue = CheckboxValue;

  created() {
    if (this.checked === null) {
      this.$emit('value-change', CheckboxValue.CHECKED);
    }
  }

  private clicked() {
    const newValue: CheckboxValue = this.checked === CheckboxValue.UNCHECKED
      ? CheckboxValue.CHECKED : CheckboxValue.UNCHECKED;
    this.$emit('value-change', newValue);
  }
}
</script>

<style scoped>
  .checkbox {
    width: 1rem;
    height: 1rem;
    border-radius: 2px;
    position: relative;
    background: var(--foreground);
    transition-duration: 0.1s;
  }

  .checked {
    background: var(--blue);
    color: var(--foreground);
    transition-duration: 0.1s;
  }

  .checkbox:hover {
    background: #e0e0e0;
    transition-duration: 0.1s;
  }

  .checkbox.checked:hover {
    background: #1B67E0;
  }

  span {
    text-align: center;
    font-weight: bold;
    font-size: 0.9rem;
    position: absolute;
  }

  .tick {
    left: 0.13rem;
    top: -0.1rem;
  }

  .dash {
    left: 0.3rem;
    top: -0.15rem;
  }
</style>
