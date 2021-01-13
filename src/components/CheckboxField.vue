<template>
  <div class="checkbox-field">
    <CheckboxInput @value-change="valueChange" :checked="checked"/>
    <div class="label">{{ label }}</div>
  </div>
</template>
<script lang="ts">
import CheckboxInput from '@/baklava/input/CheckboxInput.vue';
import { Prop, Vue, Component } from 'vue-property-decorator';
import CheckboxValue from '@/baklava/CheckboxValue';

@Component({
  components: { CheckboxInput },
})
export default class CheckboxField extends Vue {
  @Prop({ required: true }) label!: string;
  @Prop({ default: CheckboxValue.CHECKED }) checked?: CheckboxValue;

  private valueChange(newValue: CheckboxValue) {
    this.$emit('value-change', newValue);
  }
}
</script>
<style scoped>
.checkbox-field {
  display: flex;
  margin: 1rem 0;
}

.label {
  font-size: smaller;
  margin-left: 0.5rem;
  margin-top: -2px;
}
</style>
