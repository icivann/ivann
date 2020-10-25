<template>
  <div class="checkbox" :class="checked !== uncheckedValue && 'checked'" @click="clicked">
    <span v-if="checked === checkedValue" class="tick">✓</span>
    <span v-else-if="checked === halfCheckedValue" class="dash">-</span>
  </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';
import CheckboxValue from '@/baklava/CheckboxValue';

@Component({})
export default class CheckboxInput extends Vue {
  @Prop({ required: true }) checked!: CheckboxValue;

  private checkedValue = CheckboxValue.CHECKED;
  private halfCheckedValue = CheckboxValue.HALFCHECKED;
  private uncheckedValue = CheckboxValue.UNCHECKED;

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
    margin-left: 0.25rem;
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
    left: 0.15rem;
    top: -0.15rem;
  }

  .dash {
    left: 0.3rem;
    top: -0.25rem;
  }
</style>
