<template>
  <div class="d-sm-flex">
    <div class="ml-1">{{ name }}</div>
    <CheckboxInput class="ml-1" :checked="value" @value-change="changeValue"/>
  </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';
import CheckboxInput from '@/baklava/input/CheckboxInput.vue';
import CheckboxValue from '@/baklava/CheckboxValue';

@Component({
  components: { CheckboxInput },
})
export default class Checkbox extends Vue {
  @Prop() value!: CheckboxValue;
  @Prop({ type: String }) name!: string;

  private changeValue(checked: CheckboxValue) {
    this.$emit('input', checked);
  }
}
</script>
